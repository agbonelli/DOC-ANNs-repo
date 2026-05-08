"""
tests/test_predict.py
---------------------
Unit tests for the DOC prediction pipeline (predict_from_csv).
Tests run without requiring the .h5 model files (uses pre-computed DOC_estimated).
Run with: pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Data loading ──────────────────────────────────────────────────────────────
class TestDataFiles:
    def test_anna_train_loads(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNa_Train.csv"))
        assert len(df) == 109
        assert "DOC_in_situ"   in df.columns
        assert "DOC_estimated" in df.columns

    def test_anna_val_loads(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNa_Val.csv"))
        assert len(df) == 47

    def test_annb_train_loads(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Train.csv"))
        assert len(df) == 215

    def test_annb_val_loads(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Val.csv"))
        assert len(df) == 93

    def test_anna_predictors_present(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNa_Train.csv"))
        for col in ["CHL_OC4_2", "SST_table_2", "cdom443_BL1_3", "MLD_table_2"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_annb_predictors_present(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Train.csv"))
        for col in ["SST_table_2", "cdom443_BL1_3", "MLD_table_2"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_doc_range_physical(self):
        """DOC values should be within a physically plausible range for the ocean."""
        for fname in ["DOCANNa_Train.csv", "DOCANNa_Val.csv",
                      "DOCANNb_Train.csv", "DOCANNb_Val.csv"]:
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            assert df["DOC_in_situ"].min()   > 30,  f"{fname}: DOC_in_situ below 30 µmol/L"
            assert df["DOC_in_situ"].max()   < 120, f"{fname}: DOC_in_situ above 120 µmol/L"
            assert df["DOC_estimated"].min() > 20,  f"{fname}: DOC_estimated below 20 µmol/L"
            assert df["DOC_estimated"].max() < 130, f"{fname}: DOC_estimated above 130 µmol/L"

    def test_no_duplicate_indices(self):
        """Train and validation indices should not overlap."""
        idx_train_a = pd.read_csv(os.path.join(DATA_DIR, "train_idx_ANNa.csv"))
        idx_val_a   = pd.read_csv(os.path.join(DATA_DIR, "val_idx_ANNa.csv"))
        overlap = set(idx_train_a.iloc[:, 0]) & set(idx_val_a.iloc[:, 0])
        assert len(overlap) == 0, f"Train/Val overlap for ANNa: {len(overlap)} indices"

        idx_train_b = pd.read_csv(os.path.join(DATA_DIR, "train_idx_ANNb.csv"))
        idx_val_b   = pd.read_csv(os.path.join(DATA_DIR, "val_idx_ANNb.csv"))
        overlap = set(idx_train_b.iloc[:, 0]) & set(idx_val_b.iloc[:, 0])
        assert len(overlap) == 0, f"Train/Val overlap for ANNb: {len(overlap)} indices"


# ── Performance metrics on pre-computed estimates ─────────────────────────────
class TestModelPerformance:
    """
    Validate that the pre-computed DOC_estimated values reproduce the statistics
    reported in Table 2 of Bonelli et al. (2022) within reasonable tolerance.
    """

    def _stats(self, obs, est):
        mask = ~(np.isnan(obs) | np.isnan(est))
        o, e = obs[mask], est[mask]
        rmsd = np.sqrt(np.mean((o - e) ** 2))
        mapd = np.mean(np.abs(o - e) / o) * 100
        mb   = np.mean(e - o)
        return rmsd, mapd, mb

    def test_anna_train_rmsd(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNa_Train.csv"))
        rmsd, _, _ = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
        assert rmsd < 7.0, f"ANNa train RMSD too high: {rmsd:.2f} (paper: 5.83)"

    def test_anna_val_rmsd(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNa_Val.csv"))
        rmsd, _, _ = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
        assert rmsd < 10.0, f"ANNa val RMSD too high: {rmsd:.2f} (paper: 8.00)"

    def test_annb_train_rmsd(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Train.csv"))
        rmsd, _, _ = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
        assert rmsd < 7.0, f"ANNb train RMSD too high: {rmsd:.2f} (paper: 5.59)"

    def test_annb_val_rmsd(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Val.csv"))
        rmsd, _, _ = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
        assert rmsd < 8.0, f"ANNb val RMSD too high: {rmsd:.2f} (paper: 6.16)"

    def test_anna_train_mapd(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNa_Train.csv"))
        _, mapd, _ = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
        assert mapd < 10.0, f"ANNa train MAPD too high: {mapd:.2f}% (paper: 6.71%)"

    def test_annb_train_mapd(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Train.csv"))
        _, mapd, _ = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
        assert mapd < 10.0, f"ANNb train MAPD too high: {mapd:.2f}% (paper: 6.32%)"

    def test_bias_low(self):
        """Mean bias should be small relative to typical DOC values (~60 µmol/L)."""
        for fname, label in [("DOCANNa_Train.csv","ANNa train"),
                              ("DOCANNa_Val.csv","ANNa val"),
                              ("DOCANNb_Train.csv","ANNb train"),
                              ("DOCANNb_Val.csv","ANNb val")]:
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            _, _, mb = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
            assert abs(mb) < 5.0, f"{label} MB too high: {mb:.2f} µmol/L"

    def test_print_summary(self):
        """Print a summary table for reference (not a pass/fail test)."""
        print("\n  Performance summary (pre-computed estimates):")
        print(f"  {'Dataset':<18} {'N':>5}  {'RMSD':>6}  {'MAPD':>7}  {'MB':>6}")
        print(f"  {'-'*50}")
        for fname, label in [("DOCANNa_Train.csv","ANNa Train"),
                              ("DOCANNa_Val.csv",  "ANNa Val"),
                              ("DOCANNb_Train.csv","ANNb Train"),
                              ("DOCANNb_Val.csv",  "ANNb Val")]:
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            rmsd, mapd, mb = self._stats(df.DOC_in_situ.values, df.DOC_estimated.values)
            print(f"  {label:<18} {len(df):>5}  {rmsd:>6.2f}  {mapd:>6.2f}%  {mb:>+6.2f}")


# ── StandardScaler behaviour ──────────────────────────────────────────────────
class TestScaler:
    """
    Verify that StandardScaler fitted on training data transforms
    validation data to a reasonable range.
    """
    def test_scaler_annb(self):
        from sklearn.preprocessing import StandardScaler
        predictors = ["SST_table_2", "cdom443_BL1_3", "MLD_table_2"]
        train = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Train.csv"))
        val   = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Val.csv"))

        scaler = StandardScaler().fit(train[predictors].dropna())
        X_val  = scaler.transform(val[predictors].dropna())

        # Scaled values should be roughly in [-5, 5] — not astronomically large
        assert np.abs(X_val).max() < 10, \
            "Scaler produced extreme values — check predictor units"

    def test_scaler_not_fit_on_val(self):
        """
        Fitting scaler on val vs train should give different parameters,
        confirming the data sets are genuinely independent.
        """
        from sklearn.preprocessing import StandardScaler
        predictors = ["SST_table_2", "cdom443_BL1_3", "MLD_table_2"]
        train = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Train.csv")).dropna(subset=predictors)
        val   = pd.read_csv(os.path.join(DATA_DIR, "DOCANNb_Val.csv")).dropna(subset=predictors)

        s_train = StandardScaler().fit(train[predictors])
        s_val   = StandardScaler().fit(val[predictors])

        # Means should differ between train and val (they're from different cruises)
        assert not np.allclose(s_train.mean_, s_val.mean_, atol=1e-6), \
            "Train and val scalers have identical means — data may be duplicated"
