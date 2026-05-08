#!/usr/bin/env python3
"""
predict_from_netcdf.py
-----------------------
Generate a global DOC map from satellite NetCDF files using the pre-trained
DOC-ANNs model (Bonelli et al., 2022).

Full pipeline:
  1. Read SST, aCDOM(443), MLD, Chl-a from GlobColour NetCDF files
  2. Classify each pixel into one of 17 Optical Water Classes (OWC)
     from Rrs(412/443/490/510/560/670) — or load a pre-computed OWC map
  3. Route each pixel: classes 1-9 -> DOC-ANNa, classes 10-17 -> DOC-ANNb
  4. Predict DOC [umol/L] and save output as NetCDF (+ optional PNG)

Usage
-----
    # Full pipeline — compute OWC from Rrs on-the-fly (recommended):
    python predict_from_netcdf.py \
        --sst  /data/SST/L3m_20100101_SST.nc \
        --cdom /data/CDOM/L3m_20091224_CDOM443.nc \
        --mld  /data/MLD/L3m_20100101_MLD.nc \
        --chl  /data/CHL/L3m_20100101_CHL.nc \
        --rrs  /data/Rrs/L3m_20100101_412.nc \
               /data/Rrs/L3m_20100101_443.nc \
               /data/Rrs/L3m_20100101_490.nc \
               /data/Rrs/L3m_20100101_510.nc \
               /data/Rrs/L3m_20100101_560.nc \
               /data/Rrs/L3m_20100101_670.nc \
        --output outputs/DOC_20100101.nc --plot

    # With pre-computed OWC map instead of Rrs files:
    python predict_from_netcdf.py \
        --sst /data/SST/... --cdom /data/CDOM/... --mld /data/MLD/... \
        --owc /data/OWC/L3m_20100101_OWC.nc \
        --output outputs/DOC_20100101.nc

Time lag convention (Bonelli et al., 2022)
------------------------------------------
  SST, MLD, Chl-a, Rrs  ->  same 8-day composite as the target DOC date
  aCDOM(443)             ->  one 8-day composite earlier (-2 weeks lag)

Reference
---------
Bonelli, A.G., et al. (2022). Remote Sensing of Environment, 281, 113227.
https://doi.org/10.1016/j.rse.2022.113227
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

# Add repo root to path so water_classification module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from water_classification import classify_image

PREDICTORS_A = ["CHL_OC4_2", "SST_table_2", "cdom443_BL1_3", "MLD_table_2"]
PREDICTORS_B = ["SST_table_2", "cdom443_BL1_3", "MLD_table_2"]


# ---------------------------------------------------------------------------
# NetCDF I/O helpers
# ---------------------------------------------------------------------------
def read_nc_field(path):
    """Read the primary 2D variable from a GlobColour-style NetCDF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    with Dataset(path, "r") as ds:
        varnames = list(ds.variables.keys())
        lat  = np.array(ds.variables[varnames[0]][:])
        lon  = np.array(ds.variables[varnames[1]][:])
        raw  = np.array(ds.variables[varnames[2]][:].data, dtype=np.float32)
        for attr in ("_FillValue", "missing_value"):
            fv = getattr(ds.variables[varnames[2]], attr, None)
            if fv is not None:
                raw[np.abs(raw - float(fv)) < 1e-3 * abs(float(fv)) + 1] = np.nan
        raw[raw > 1e10]  = np.nan
        raw[raw < -1e5]  = np.nan
    return lat, lon, raw


def save_nc(path, lat, lon, doc, owc, model_used):
    """Save DOC and OWC maps to a NetCDF4 file."""
    ds = Dataset(path, "w", format="NETCDF4_CLASSIC")
    ds.title      = "Surface DOC estimated by DOC-ANNs (Bonelli et al., 2022)"
    ds.reference  = "doi:10.1016/j.rse.2022.113227"
    ds.history    = f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    ds.model      = model_used

    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))

    v_lat = ds.createVariable("lat",  np.float32, ("lat",))
    v_lon = ds.createVariable("lon",  np.float32, ("lon",))
    v_doc = ds.createVariable("DOC",  np.float32, ("lat", "lon"), fill_value=np.float32(1e20))
    v_owc = ds.createVariable("OWC",  np.float32, ("lat", "lon"), fill_value=np.float32(1e20))

    v_lat.units = "degrees_north"
    v_lon.units = "degrees_east"
    v_doc.units = "umol/L"
    v_doc.long_name = "Surface dissolved organic carbon concentration"
    v_owc.units = "1"
    v_owc.long_name = "Optical Water Class (1-9: coastal/ANNa, 10-17: open ocean/ANNb)"

    v_lat[:] = lat
    v_lon[:] = lon
    v_doc[:, :] = np.where(np.isnan(doc), 1e20, doc.astype(np.float32))
    if owc is not None:
        v_owc[:, :] = np.where(np.isnan(owc), 1e20, owc.astype(np.float32))
    ds.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def estimate_doc(sst, cdom, mld, chl, owc, model_a, model_b, scaler_a, scaler_b):
    """
    Estimate DOC for every valid pixel using the OWC-based switch.

    - Classes 1-9  (coastal)    -> DOC-ANNa  (requires Chl-a)
    - Classes 10-17 (open ocean) -> DOC-ANNb
    - owc is None               -> DOC-ANNb applied everywhere
    """
    shape = sst.shape
    N     = shape[0] * shape[1]
    doc   = np.full(N, np.nan, dtype=np.float32)

    sst_f  = sst.reshape(N)
    cdom_f = cdom.reshape(N)
    mld_f  = mld.reshape(N)
    chl_f  = chl.reshape(N)  if chl  is not None else None
    owc_f  = owc.reshape(N).astype(float) if owc is not None else None

    if owc_f is not None:
        mask_b = (owc_f >= 10) & (owc_f <= 17)
        mask_a = (owc_f >= 1)  & (owc_f <= 9)
    else:
        mask_b = np.ones(N, dtype=bool)
        mask_a = np.zeros(N, dtype=bool)

    # DOC-ANNb (open ocean)
    if model_b is not None and mask_b.any():
        df_b = pd.DataFrame({
            "SST_table_2":   sst_f[mask_b],
            "cdom443_BL1_3": cdom_f[mask_b],
            "MLD_table_2":   mld_f[mask_b],
        })
        valid = df_b.dropna()
        if len(valid) > 0:
            X = scaler_b.transform(valid.values)
            pDOC = np.squeeze(model_b.predict(X, verbose=0, batch_size=4096))
            idx = np.where(mask_b)[0][valid.index]
            doc[idx] = pDOC
        print(f"  DOC-ANNb: {mask_b.sum():,} pixels, {len(valid):,} valid")

    # DOC-ANNa (coastal) — needs Chl-a
    if model_a is not None and mask_a.any():
        if chl_f is None:
            print("  DOC-ANNa skipped: --chl file not provided.")
        else:
            df_a = pd.DataFrame({
                "CHL_OC4_2":     chl_f[mask_a],
                "SST_table_2":   sst_f[mask_a],
                "cdom443_BL1_3": cdom_f[mask_a],
                "MLD_table_2":   mld_f[mask_a],
            })
            valid = df_a.dropna()
            if len(valid) > 0:
                X = scaler_a.transform(valid.values)
                pDOC = np.squeeze(model_a.predict(X, verbose=0, batch_size=4096))
                idx = np.where(mask_a)[0][valid.index]
                doc[idx] = pDOC
            print(f"  DOC-ANNa: {mask_a.sum():,} pixels, {len(valid):,} valid")

    return doc.reshape(shape)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_doc_map(doc, owc, lat, lon, output_path):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    has_owc = owc is not None
    ncols = 2 if has_owc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14 * ncols // 2 + 7, 6))
    if ncols == 1:
        axes = [axes]

    # DOC map
    ax = axes[0]
    im = ax.pcolormesh(lon, lat, doc, cmap="viridis",
                       norm=mcolors.Normalize(vmin=40, vmax=85),
                       shading="auto", rasterized=True)
    fig.colorbar(im, ax=ax, orientation="horizontal",
                 fraction=0.032, pad=0.06).set_label("DOC [µmol L⁻¹]")
    ax.set_title("Surface DOC — DOC-ANNs (Bonelli et al., 2022)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # OWC map
    if has_owc:
        ax2 = axes[1]
        cmap_owc = plt.cm.get_cmap("tab20", 17)
        im2 = ax2.pcolormesh(lon, lat, owc, cmap=cmap_owc,
                             norm=mcolors.Normalize(vmin=0.5, vmax=17.5),
                             shading="auto", rasterized=True)
        cb2 = fig.colorbar(im2, ax=ax2, orientation="horizontal",
                           fraction=0.032, pad=0.06, ticks=range(1, 18))
        cb2.set_label("OWC class (1-9: ANNa | 10-17: ANNb)")
        ax2.set_title("Optical Water Class")
        ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict global surface DOC from satellite NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sst",  required=True, help="NetCDF: SST (same date as DOC)")
    p.add_argument("--cdom", required=True, help="NetCDF: aCDOM(443) (-2 weeks lag)")
    p.add_argument("--mld",  required=True, help="NetCDF: MLD (same date as DOC)")
    p.add_argument("--chl",  default=None,  help="NetCDF: Chl-a (ANNa only)")

    owc_group = p.add_mutually_exclusive_group()
    owc_group.add_argument("--rrs", nargs=6,
                           metavar=("RRS412","RRS443","RRS490","RRS510","RRS560","RRS670"),
                           default=None,
                           help="6 NetCDF Rrs files at 412/443/490/510/560/670 nm — "
                                "used to compute OWC on-the-fly (recommended)")
    owc_group.add_argument("--owc", default=None,
                           help="NetCDF: pre-computed OWC class map")

    p.add_argument("--lut_dir",
                   default=os.path.join(os.path.dirname(__file__), "..", "models", "OWC_LUT"),
                   help="OWC LUT directory (default: models/OWC_LUT)")
    p.add_argument("--output", required=True, help="Output NetCDF path")
    p.add_argument("--model_a",
                   default=os.path.join(os.path.dirname(__file__), "..", "models", "DOCANNa.h5"))
    p.add_argument("--model_b",
                   default=os.path.join(os.path.dirname(__file__), "..", "models", "DOCANNb.h5"))
    p.add_argument("--scaler_a",
                   default=os.path.join(os.path.dirname(__file__), "..", "models", "DOCANNa_scaler.pkl"))
    p.add_argument("--scaler_b",
                   default=os.path.join(os.path.dirname(__file__), "..", "models", "DOCANNb_scaler.pkl"))
    p.add_argument("--plot", action="store_true", help="Save a PNG map alongside the NetCDF")
    return p.parse_args()


def main():
    args = parse_args()
    import joblib
    import tensorflow as tf

    # ---- Load models ----
    print("Loading models...")
    models = {}
    for name, path in [("ANNa", args.model_a), ("ANNb", args.model_b)]:
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path)
            print(f"  Loaded DOC-{name}")
        else:
            models[name] = None
            print(f"  DOC-{name} not found at {path} — skipping")

    if models["ANNb"] is None and models["ANNa"] is None:
        print("Error: no model files found.", file=sys.stderr); sys.exit(1)

    # ---- Load scalers ----
    print("Loading scalers...")
    scalers = {}
    for name, path in [("ANNa", args.scaler_a), ("ANNb", args.scaler_b)]:
        if os.path.exists(path):
            scalers[name] = joblib.load(path)
            print(f"  Loaded scaler for DOC-{name}")
        else:
            print(f"  Warning: scaler for DOC-{name} not found at {path}")
            scalers[name] = None

    # ---- Read satellite fields ----
    print("\nReading satellite fields...")
    lat, lon, sst  = read_nc_field(args.sst)
    _,   _,   cdom = read_nc_field(args.cdom)
    _,   _,   mld  = read_nc_field(args.mld)
    chl = None
    if args.chl:
        _, _, chl = read_nc_field(args.chl)

    # ---- OWC classification ----
    owc = None
    if args.rrs:
        print("  Computing OWC from Rrs bands...")
        rrs_bands = [read_nc_field(f)[2] for f in args.rrs]
        owc = classify_image(rrs_bands, lut_dir=args.lut_dir)
        n_a = int(np.nansum((owc >= 1)  & (owc <= 9)))
        n_b = int(np.nansum((owc >= 10) & (owc <= 17)))
        print(f"  OWC done — ANNa (1-9): {n_a:,} px  |  ANNb (10-17): {n_b:,} px")
    elif args.owc:
        _, _, owc = read_nc_field(args.owc)
        print(f"  OWC loaded from file")
    else:
        print("  No OWC input — applying DOC-ANNb to all ocean pixels")

    print(f"  Grid: {len(lat)} x {len(lon)}  |  "
          f"SST: {np.nanmin(sst):.1f}–{np.nanmax(sst):.1f} °C  |  "
          f"CDOM: {np.nanmin(cdom):.4f}–{np.nanmax(cdom):.4f} m⁻¹")

    # ---- Estimate DOC ----
    print("\nEstimating DOC...")
    t0 = time.time()
    doc = estimate_doc(sst, cdom, mld, chl, owc,
                       models["ANNa"], models["ANNb"],
                       scalers["ANNa"], scalers["ANNb"])
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"{np.sum(~np.isnan(doc)):,} valid pixels  |  "
          f"DOC: {np.nanmin(doc):.1f}–{np.nanmax(doc):.1f} µmol/L")

    # ---- Save ----
    print("\nSaving output...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    model_str = ("DOC-ANNs (ANNa+ANNb, OWC switch)" if owc is not None
                 else "DOC-ANNb (open ocean, no OWC)")
    save_nc(args.output, lat, lon, doc, owc, model_str)

    if args.plot:
        plot_doc_map(doc, owc, lat, lon,
                     os.path.splitext(args.output)[0] + ".png")

    print("\nDone.")


if __name__ == "__main__":
    main()
