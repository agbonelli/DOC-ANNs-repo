#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_DOCNNs.py
-------------
Estimate surface DOC [umol/L] from GlobColour satellite images and save
one NetCDF map per 8-day composite period.

Filename conventions
--------------------
  SST / CDOM / MLD / CHL   (each in its own subfolder):
    L3m_[YYYYMMDD]-[YYYYMMDD]__GLOB_25_GSM-[Sensor]_[VAR]_8D_00.nc
    e.g.: L3m_20100101-20100108__GLOB_25_GSM-MODVIR_SST_8D_00.nc

  Rrs   (in RRS/):
    L3m_[YYYYMMDD]-[YYYYMMDD]__GLOB_25_[Sensor]_NRRS[WL]_8D_00.nc
    e.g.: L3m_20100101-20100108__GLOB_25_MODVIR_NRRS443_8D_00.nc

Expected folder structure
-------------------------
  satellite_data/
    SST/              L3m_*_SST_8D_00.nc
    CDOM443BL1/       L3m_*_CDOM443BL1_8D_00.nc
    MLD/              L3m_*_MLD_8D_00.nc
    CHL1/             L3m_*_CHL1_8D_00.nc
    RRS/    L3m_*_NRRS412_8D_00.nc
                      L3m_*_NRRS443_8D_00.nc
                      L3m_*_NRRS490_8D_00.nc
                      L3m_*_NRRS510_8D_00.nc
                      L3m_*_NRRS560_8D_00.nc
                      L3m_*_NRRS670_8D_00.nc

Temporal lag convention (Bonelli et al., 2022)
----------------------------------------------
  Variable        lag_periods   Meaning
  ────────────────────────────────────────────────────────────────
  SST             0             one composite earlier  (-8 days)
  MLD             0             one composite earlier  (-8 days)
  CHL1 (ANNa)     0             one composite earlier  (-8 days)
  Rrs (OWC)       0             same 8-day composite as DOC target
  aCDOM(443)      1             two composite earlier  (-16 days)

Usage
-----
    python run_DOCNNs.py            # edit CONFIGURATION block first
    docker compose run --rm cli run_DOCNNs.py

Reference
---------
Bonelli, A.G., et al. (2022). Remote Sensing of Environment, 281, 113227.
https://doi.org/10.1016/j.rse.2022.113227
"""

import os
import re
import sys
import glob
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────

# Base directory: the repo root (where this script lives)
_REPO = os.path.dirname(os.path.abspath(__file__))

# Satellite data folders — defaults point to satellite_data/ inside the repo.
# If your data is elsewhere, replace with absolute paths, e.g.:
#   DIR_SST = "/Volumes/MyDisk/GlobColour/SST"
DIR_SST  = os.path.join(_REPO, "satellite_data", "SST")
DIR_CDOM = os.path.join(_REPO, "satellite_data", "CDOM443BL1")
DIR_MLD  = os.path.join(_REPO, "satellite_data", "MLD")
DIR_CHL  = os.path.join(_REPO, "satellite_data", "CHL1")
DIR_RRS  = os.path.join(_REPO, "satellite_data", "RRS")

# Variable tokens as they appear in filenames
VAR_SST  = "SST"
VAR_CDOM = "CDOM443BL1"
VAR_MLD  = "MLD"
VAR_CHL  = "CHL1"
VAR_RRS  = {412: "NRRS412", 443: "NRRS443", 490: "NRRS490",
            510: "NRRS510", 560: "NRRS560", 670: "NRRS670"}

# Model and scaler paths (relative to repo root)
MODEL_ANNa  = os.path.join(_REPO, "models", "DOCANNa.h5")
MODEL_ANNb  = os.path.join(_REPO, "models", "DOCANNb.h5")
SCALER_ANNa = os.path.join(_REPO, "models", "DOCANNa_scaler.pkl")
SCALER_ANNb = os.path.join(_REPO, "models", "DOCANNb_scaler.pkl")
LUT_DIR     = os.path.join(_REPO, "models", "OWC_LUT")

# Output folder for DOC NetCDF maps
OUTPUT_DIR = os.path.join(_REPO, "satellite_data", "DOC_output")

# Toggle models
RUN_ANNa = True   # coastal (needs CHL + Rrs for OWC)
RUN_ANNb = True   # open ocean

# ─────────────────────────────────────────────────────────────────────────────
# FILENAME PARSING
# Two patterns to handle the slight format difference between SST/CDOM/MLD/CHL
# and Rrs files (GSM-Sensor vs plain Sensor, VAR vs NRRSXXX).
# ─────────────────────────────────────────────────────────────────────────────

# Pattern for SST / CDOM443BL1 / MLD / CHL1
#   L3m_20100101-20100108__GLOB_25_GSM-MODVIR_SST_8D_00.nc
_RE_STANDARD = re.compile(
    r"L3m_(\d{8})-(\d{8})__GLOB_25_(?:GSM-)?[^_]+_([A-Z0-9]+)_8D_\d+\.nc$",
    re.IGNORECASE,
)

# Pattern for Rrs  (no GSM- prefix, variable is NRRS412 etc.)
#   L3m_20100101-20100108__GLOB_25_MODVIR_NRRS443_8D_00.nc
_RE_RRS = re.compile(
    r"L3m_(\d{8})-(\d{8})__GLOB_25_[^_]+_(NRRS\d{3})_8D_\d+\.nc$",
    re.IGNORECASE,
)


def parse_filename(path):
    """
    Extract (date_start, date_end, variable_token) from a GlobColour filename.
    Returns None if the filename does not match either expected pattern.
    """
    fname = os.path.basename(path)
    for pattern in (_RE_RRS, _RE_STANDARD):
        m = pattern.match(fname)
        if m:
            return (
                datetime.strptime(m.group(1), "%Y%m%d"),
                datetime.strptime(m.group(2), "%Y%m%d"),
                m.group(3).upper(),
            )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG BUILDING — one dict per variable folder
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(folder, var_token):
    """
    Scan `folder` for files matching var_token and return a dict:
        {date_start: filepath}
    """
    index = {}
    if not os.path.isdir(folder):
        return index
    for fpath in sorted(glob.glob(os.path.join(folder, "L3m_*.nc"))):
        result = parse_filename(fpath)
        if result is None:
            continue
        d_start, _, var = result
        if var == var_token.upper():
            index[d_start] = fpath
    return index


def build_catalogs():
    """Build one catalog dict per variable from the configured folders."""
    catalogs = {
        VAR_SST:  scan_folder(DIR_SST,  VAR_SST),
        VAR_CDOM: scan_folder(DIR_CDOM, VAR_CDOM),
        VAR_MLD:  scan_folder(DIR_MLD,  VAR_MLD),
        VAR_CHL:  scan_folder(DIR_CHL,  VAR_CHL),
    }
    for wl, token in VAR_RRS.items():
        catalogs[token] = scan_folder(DIR_RRS, token)
    return catalogs


# ─────────────────────────────────────────────────────────────────────────────
# LAG RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def find_lagged_file(catalog, var_token, target_date, lag_periods):
    """
    Return the filepath for var_token whose start date is
    lag_periods * 8 days before target_date.

    lag_periods=0  →  same period (start date == target_date)
    lag_periods=1  →  one period earlier (start date == target_date - 8 days)
    lag_periods=2  →  two periods earlier (start date == target_date - 16 days)
    """
    lagged = target_date - timedelta(days=8 * lag_periods)
    return catalog.get(var_token, {}).get(lagged, None)


def available_dates(catalogs, var_token):
    """Sorted list of start dates available for var_token."""
    return sorted(catalogs.get(var_token, {}).keys())


# ─────────────────────────────────────────────────────────────────────────────
# NetCDF I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_nc(path):
    """
    Read the primary 2D field from a GlobColour NetCDF file.
    Returns (lat, lon, data_2d) — fill values replaced by NaN.
    """
    with Dataset(path, "r") as ds:
        keys = list(ds.variables.keys())
        lat  = np.array(ds.variables[keys[0]][:], dtype=np.float32)
        lon  = np.array(ds.variables[keys[1]][:], dtype=np.float32)
        raw  = np.ma.filled(
            ds.variables[keys[2]][:].astype(np.float32), fill_value=np.nan
        )
        raw[raw > 1e10]  = np.nan
        raw[raw < -1e5]  = np.nan
    return lat, lon, raw


def save_nc(path, lat, lon, doc_map, owc_map, date_start, date_end):
    """Save DOC (and OWC when available) to a NetCDF4 file."""
    ds = Dataset(path, "w", format="NETCDF4_CLASSIC")
    ds.title     = "Surface DOC estimated by DOC-ANNs"
    ds.reference = "Bonelli et al. (2022) doi:10.1016/j.rse.2022.113227"
    ds.period    = (f"{date_start.strftime('%Y-%m-%d')} to "
                    f"{date_end.strftime('%Y-%m-%d')}")

    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))

    v_lat           = ds.createVariable("lat", np.float32, ("lat",))
    v_lon           = ds.createVariable("lon", np.float32, ("lon",))
    v_doc           = ds.createVariable("DOC", np.float32, ("lat","lon"),
                                        fill_value=np.float32(1e20))
    v_lat.units     = "degrees_north"
    v_lon.units     = "degrees_east"
    v_doc.units     = "umol/L"
    v_doc.long_name = "Surface dissolved organic carbon"

    v_lat[:] = lat
    v_lon[:] = lon
    v_doc[:] = np.where(np.isnan(doc_map), 1e20, doc_map)

    if owc_map is not None:
        v_owc           = ds.createVariable("OWC", np.float32, ("lat","lon"),
                                            fill_value=np.float32(1e20))
        v_owc.units     = "1"
        v_owc.long_name = "Optical Water Class (1-9: ANNa, 10-17: ANNb)"
        v_owc[:]        = np.where(np.isnan(owc_map), 1e20, owc_map)

    ds.close()


# ─────────────────────────────────────────────────────────────────────────────
# DOC ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_doc(fields, owc_map, model_a, model_b, scaler_a, scaler_b, shape):
    """
    Predict DOC for every valid pixel.
    fields dict keys: 'sst', 'cdom', 'mld', optionally 'chl'.
    """
    N   = shape[0] * shape[1]
    doc = np.full(N, np.nan, dtype=np.float32)

    sst_f  = fields["sst"].reshape(N)
    cdom_f = fields["cdom"].reshape(N)
    mld_f  = fields["mld"].reshape(N)
    chl_f  = fields["chl"].reshape(N) if "chl" in fields else None
    owc_f  = owc_map.reshape(N)       if owc_map is not None else None

    if owc_f is not None:
        mask_b = (owc_f >= 10) & (owc_f <= 17)
        mask_a = (owc_f >= 1)  & (owc_f <= 9)
    else:
        mask_b = np.ones(N, dtype=bool)
        mask_a = np.zeros(N, dtype=bool)

    # DOC-ANNb (open ocean)
    # Models were saved with input_shape=(None, None, n_features) so they expect
    # 3D input: (batch, 1, features). We add the middle dimension with np.newaxis.
    if model_b is not None and mask_b.any():
        df = pd.DataFrame({
            "SST":  sst_f[mask_b],
            "CDOM": cdom_f[mask_b],
            "MLD":  mld_f[mask_b],
        }).dropna()
        if len(df) > 0:
            X    = scaler_b.transform(df.values)[:, np.newaxis, :]  # (N,3) → (N,1,3)
            pred = np.squeeze(model_b.predict(X, verbose=0, batch_size=8192))
            doc[np.where(mask_b)[0][df.index]] = pred

    # DOC-ANNa (coastal)
    if model_a is not None and mask_a.any() and chl_f is not None:
        df = pd.DataFrame({
            "CHL":  chl_f[mask_a],
            "SST":  sst_f[mask_a],
            "CDOM": cdom_f[mask_a],
            "MLD":  mld_f[mask_a],
        }).dropna()
        if len(df) > 0:
            X    = scaler_a.transform(df.values)[:, np.newaxis, :]  # (N,4) → (N,1,4)
            pred = np.squeeze(model_a.predict(X, verbose=0, batch_size=8192))
            doc[np.where(mask_a)[0][df.index]] = pred

    return doc.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import tensorflow as tf
    import joblib
    from water_classification import load_lut, classify_image

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load models and scalers ──────────────────────────────────────────────
    print("Loading models and scalers...")
    model_a = scaler_a = model_b = scaler_b = None

    if RUN_ANNa:
        if os.path.exists(MODEL_ANNa):
            model_a  = tf.keras.models.load_model(MODEL_ANNa)
            scaler_a = joblib.load(SCALER_ANNa)
            print("  DOC-ANNa loaded")
        else:
            print(f"  WARNING: DOC-ANNa not found at {MODEL_ANNa}")

    if RUN_ANNb:
        if os.path.exists(MODEL_ANNb):
            model_b  = tf.keras.models.load_model(MODEL_ANNb)
            scaler_b = joblib.load(SCALER_ANNb)
            print("  DOC-ANNb loaded")
        else:
            print(f"  WARNING: DOC-ANNb not found at {MODEL_ANNb}")

    if model_a is None and model_b is None:
        print("ERROR: No models available. Check model paths in CONFIGURATION.")
        sys.exit(1)

    # ── Load OWC LUTs ────────────────────────────────────────────────────────
    print("Loading OWC LUTs...")
    mu, sigma = load_lut(LUT_DIR)
    print(f"  {mu.shape[0]} classes loaded")

    # ── Build file catalogs ──────────────────────────────────────────────────
    print("\nScanning satellite data folders...")
    catalogs = build_catalogs()

    for var, idx in catalogs.items():
        if idx:
            dates = sorted(idx.keys())
            print(f"  {var:<14}: {len(idx):>4} files  "
                  f"({dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')})")
        else:
            print(f"  {var:<14}: 0 files  — check DIR_{var} in CONFIGURATION")

    # ── Reference timeline from SST ──────────────────────────────────────────
    ref_dates = available_dates(catalogs, VAR_SST)
    if not ref_dates:
        print(f"\nERROR: No SST files found in {DIR_SST}")
        sys.exit(1)

    print(f"\n{len(ref_dates)} periods to process  "
          f"({ref_dates[0].strftime('%Y-%m-%d')} → {ref_dates[-1].strftime('%Y-%m-%d')})")
    print("Lag convention:")
    print("  SST, MLD, CHL1, Rrs  →  lag=0  (same 8-day period)")
    print("  CDOM443BL1           →  lag=1  (one period earlier, -8 days)")

    # ── Process each period ──────────────────────────────────────────────────
    n_ok = n_skip = n_exist = 0

    for date_start in ref_dates:
        date_end = date_start + timedelta(days=7)

        out_fname = (f"DOC_{date_start.strftime('%Y%m%d')}-"
                     f"{date_end.strftime('%Y%m%d')}_8D.nc")
        out_path  = os.path.join(OUTPUT_DIR, out_fname)

        if os.path.exists(out_path):
            n_exist += 1
            continue

        # ── Resolve files with correct lags ──────────────────────────────
        f_sst  = find_lagged_file(catalogs, VAR_SST,  date_start, lag_periods=1)
        f_cdom = find_lagged_file(catalogs, VAR_CDOM, date_start, lag_periods=2)
        f_mld  = find_lagged_file(catalogs, VAR_MLD,  date_start, lag_periods=1)
        f_chl  = find_lagged_file(catalogs, VAR_CHL,  date_start, lag_periods=1)
        f_rrs  = {wl: find_lagged_file(catalogs, token, date_start, lag_periods=0)
                  for wl, token in VAR_RRS.items()}

        # Minimum required: SST + CDOM + MLD
        missing = [v for v, f in [(VAR_SST, f_sst),
                                   (VAR_CDOM, f_cdom),
                                   (VAR_MLD, f_mld)] if not f]
        if missing:
            print(f"  {date_start.strftime('%Y-%m-%d')}  SKIP — missing: {missing}")
            n_skip += 1
            continue

        # ── Read satellite fields ─────────────────────────────────────────
        lat, lon, sst  = read_nc(f_sst)
        _,   _,   cdom = read_nc(f_cdom)
        _,   _,   mld  = read_nc(f_mld)
        shape = sst.shape

        fields = {"sst": sst, "cdom": cdom, "mld": mld}
        if f_chl:
            _, _, chl = read_nc(f_chl)
            fields["chl"] = chl

        # ── OWC classification ────────────────────────────────────────────
        owc_map = None
        have_rrs = all(f_rrs[wl] is not None for wl in f_rrs)
        if have_rrs:
            rrs_bands = [read_nc(f_rrs[wl])[2]
                         for wl in [412, 443, 490, 510, 560, 670]]
            owc_map   = classify_image(rrs_bands, lut_dir=LUT_DIR)
            n_a = int(np.nansum((owc_map >= 1)  & (owc_map <= 9)))
            n_b = int(np.nansum((owc_map >= 10) & (owc_map <= 17)))
        else:
            n_a, n_b = 0, int(np.nansum(~np.isnan(sst)))
            if model_a is not None and RUN_ANNa:
                print(f"  {date_start.strftime('%Y-%m-%d')}  "
                      f"WARNING: Rrs missing — ANNa skipped, ANNb applied everywhere")

        # ── Estimate DOC ──────────────────────────────────────────────────
        doc_map  = estimate_doc(fields, owc_map,
                                model_a, model_b,
                                scaler_a, scaler_b, shape)
        n_valid  = int(np.nansum(~np.isnan(doc_map)))
        doc_mean = float(np.nanmean(doc_map)) if n_valid > 0 else np.nan

        # ── Save ──────────────────────────────────────────────────────────
        save_nc(out_path, lat, lon, doc_map, owc_map, date_start, date_end)

        print(f"  {date_start.strftime('%Y-%m-%d')}  "
              f"ANNa={n_a:>7,}px  ANNb={n_b:>7,}px  "
              f"valid={n_valid:>7,}  mean={doc_mean:5.1f} µmol/L  "
              f"→ {out_fname}")
        n_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Finished.")
    print(f"  Processed : {n_ok:>5}")
    print(f"  Skipped   : {n_skip:>5}  (missing input files)")
    print(f"  Already OK: {n_exist:>5}  (output existed, not reprocessed)")
    print(f"  Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
