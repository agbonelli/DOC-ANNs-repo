#!/usr/bin/env python3
"""
predict_from_csv.py
-------------------
Predict surface DOC [µmol/L] from tabular data using the pre-trained DOC-ANNs models.

This is the simplest way to use the DOC-ANNs models: provide a CSV file with the
required predictor columns, and the script outputs a CSV with the estimated DOC column
appended.

Usage
-----
    python predict_from_csv.py --model ANNb --input my_data.csv --output my_data_with_DOC.csv

    # Use ANNa (coastal waters) with a custom model path:
    python predict_from_csv.py --model ANNa --input coastal_data.csv \\
        --model_path /path/to/DOCANNa.h5

Input CSV format
----------------
For DOC-ANNb (open ocean):
    SST_table_2, cdom443_BL1_3, MLD_table_2

For DOC-ANNa (coastal / optically complex waters):
    CHL_OC4_2, SST_table_2, cdom443_BL1_3, MLD_table_2

The column names must match exactly (same convention as the matchup database).
Rows with any NaN in the required columns are skipped.

Reference
---------
Bonelli, A.G., Loisel, H., Jorge, D.S.F., Mangin, A., Fanton d'Andon, O.,
& Vantrepotte, V. (2022). A new method to estimate the dissolved organic carbon
concentration from remote sensing in the global open ocean.
Remote Sensing of Environment, 281, 113227.
https://doi.org/10.1016/j.rse.2022.113227
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Predictor definitions — must match the column names in the matchup database
# ---------------------------------------------------------------------------
PREDICTORS = {
    "ANNa": ["CHL_OC4_2", "SST_table_2", "cdom443_BL1_3", "MLD_table_2"],
    "ANNb": ["SST_table_2", "cdom443_BL1_3", "MLD_table_2"],
}

DEFAULT_MODEL_PATHS = {
    "ANNa": os.path.join(os.path.dirname(__file__), "..", "models", "DOCANNa.h5"),
    "ANNb": os.path.join(os.path.dirname(__file__), "..", "models", "DOCANNb.h5"),
}


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------
def predict_doc(df: pd.DataFrame, model_name: str, model_path: str) -> pd.Series:
    """
    Predict DOC [µmol/L] for each row in df.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain the columns required by model_name.
    model_name : str
        "ANNa" or "ANNb".
    model_path : str
        Path to the .h5 Keras model file.

    Returns
    -------
    pd.Series
        Predicted DOC values (NaN for rows with missing inputs).
    """
    import tensorflow as tf

    predictors = PREDICTORS[model_name]
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise ValueError(
            f"DOC-{model_name} requires columns {predictors}.\n"
            f"Missing in input file: {missing}"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Request the pre-trained weights at abonelli@asu.edu "
            "or see the README for instructions."
        )

    model = tf.keras.models.load_model(model_path)
    print(f"  Loaded DOC-{model_name} from {model_path}")

    # Work only on rows with complete data
    valid = df[predictors].dropna()
    idx = valid.index.tolist()

    result = pd.Series(np.nan, index=df.index, name="DOC_estimated")

    if len(valid) == 0:
        print("  Warning: no valid rows found (all NaN).")
        return result

    X = StandardScaler().fit_transform(valid.values)
    pDOC = np.squeeze(model.predict(X, verbose=0))
    result.loc[idx] = pDOC

    return result


# ---------------------------------------------------------------------------
# Demo mode — runs without real model files or data
# ---------------------------------------------------------------------------
def run_demo(model_name: str) -> None:
    """Load the included validation CSV and print a quick summary."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    fname = f"DOCAN{model_name[-1]}_Val.csv"
    path = os.path.join(data_dir, fname)

    if not os.path.exists(path):
        print(f"Demo file not found: {path}")
        return

    df = pd.read_csv(path)
    predictors = PREDICTORS[model_name]
    model_path = DEFAULT_MODEL_PATHS[model_name]

    print(f"\nRunning demo with: {fname}  ({len(df)} rows)")
    print(f"Predictors: {predictors}")

    if os.path.exists(model_path):
        pDOC = predict_doc(df, model_name, model_path)
    else:
        # Stub for demo: use the pre-computed DOC_estimated column
        print(f"  Model not found — using pre-computed DOC_estimated column from {fname}")
        pDOC = df["DOC_estimated"]

    obs = df["DOC_in_situ"].values
    est = pDOC.values
    mask = ~(np.isnan(obs) | np.isnan(est))
    obs, est = obs[mask], est[mask]

    rmsd = np.sqrt(np.mean((obs - est) ** 2))
    mapd = np.mean(np.abs(obs - est) / obs) * 100
    mb = np.mean(est - obs)
    r2 = np.corrcoef(obs, est)[0, 1] ** 2

    print(f"\n  DOC-{model_name} validation statistics")
    print(f"  ─────────────────────────────────────")
    print(f"  N     = {len(obs)}")
    print(f"  RMSD  = {rmsd:.2f} µmol/L")
    print(f"  MAPD  = {mapd:.2f} %")
    print(f"  MB    = {mb:.2f} µmol/L")
    print(f"  R²    = {r2:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict surface DOC [µmol/L] from tabular data using DOC-ANNs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", choices=["ANNa", "ANNb"], default="ANNb",
        help="Which sub-model to use. ANNb = open ocean (default), ANNa = coastal waters.",
    )
    p.add_argument(
        "--input", default=None,
        help="Path to input CSV file. If omitted, runs a demo using the validation dataset.",
    )
    p.add_argument(
        "--output", default=None,
        help="Path for the output CSV (input + DOC_estimated column). "
             "Default: input file with '_DOC' suffix.",
    )
    p.add_argument(
        "--model_path", default=None,
        help="Path to the .h5 model file. Default: models/DOCAN{a,b}.h5",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    model_path = args.model_path or DEFAULT_MODEL_PATHS[model_name]

    if args.input is None:
        # ---- Demo mode ----
        print("No input file provided — running demo with validation dataset.")
        print("Use --input your_data.csv to predict on your own data.\n")
        run_demo(model_name)
        return

    # ---- Normal mode ----
    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    pDOC = predict_doc(df, model_name, model_path)
    df["DOC_estimated"] = pDOC

    n_valid = pDOC.notna().sum()
    print(f"  DOC estimated for {n_valid}/{len(df)} rows")
    print(f"  DOC range: {pDOC.min():.1f} – {pDOC.max():.1f} µmol/L")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_DOC" + (ext or ".csv")
    else:
        out_path = args.output

    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
