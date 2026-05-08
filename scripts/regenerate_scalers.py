#!/usr/bin/env python3
"""
scripts/regenerate_scalers.py
------------------------------
Regenerate the StandardScaler .pkl files from the training CSVs.

Run this INSIDE the Docker container to ensure the scalers are serialised
with the same numpy/joblib versions used at inference time:

    docker compose run --rm cli scripts/regenerate_scalers.py

The scalers are saved to models/DOCANNa_scaler.pkl and models/DOCANNb_scaler.pkl.
These files must be present before running run_DOCNNs.py or predict_from_csv.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

PREDICTORS_A = ["CHL_OC4_2", "SST_table_2", "cdom443_BL1_3", "MLD_table_2"]
PREDICTORS_B = ["SST_table_2", "cdom443_BL1_3", "MLD_table_2"]


def make_scaler(csv_path, predictors, out_path):
    df = pd.read_csv(csv_path)
    X  = df[predictors].dropna().values

    scaler = StandardScaler()
    scaler.fit(X)

    joblib.dump(scaler, out_path)

    print(f"  Saved: {out_path}")
    print(f"    N        = {len(X)}")
    print(f"    mean     = {scaler.mean_.round(4)}")
    print(f"    std      = {scaler.scale_.round(4)}")
    print(f"    numpy    = {np.__version__}")
    print(f"    joblib   = {joblib.__version__}")
    print(f"    sklearn  = {__import__('sklearn').__version__}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Regenerating DOC-ANNa scaler...")
    make_scaler(
        csv_path   = os.path.join(DATA_DIR, "DOCANNa_Train.csv"),
        predictors = PREDICTORS_A,
        out_path   = os.path.join(MODELS_DIR, "DOCANNa_scaler.pkl"),
    )

    print("\nRegenerating DOC-ANNb scaler...")
    make_scaler(
        csv_path   = os.path.join(DATA_DIR, "DOCANNb_Train.csv"),
        predictors = PREDICTORS_B,
        out_path   = os.path.join(MODELS_DIR, "DOCANNb_scaler.pkl"),
    )

    print("\nDone. Both scalers saved to models/")


if __name__ == "__main__":
    main()
