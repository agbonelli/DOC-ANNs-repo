"""
tests/test_owc.py
-----------------
Unit tests for the Optical Water Class (OWC) classifier.
Run with: pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from water_classification import load_lut, classify, classify_image, normalise_rrs, owc_to_doc_model

LUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "OWC_LUT")


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def lut():
    mu, sigma = load_lut(LUT_DIR)
    return mu, sigma


# ── LUT loading ───────────────────────────────────────────────────────────────
class TestLoadLUT:
    def test_shapes(self, lut):
        mu, sigma = lut
        assert mu.shape    == (17, 6), f"Expected (17,6), got {mu.shape}"
        assert sigma.shape == (17, 6, 6), f"Expected (17,6,6), got {sigma.shape}"

    def test_covariance_positive_definite(self, lut):
        """All covariance matrices must have positive eigenvalues."""
        _, sigma = lut
        for i in range(17):
            eigvals = np.linalg.eigvalsh(sigma[i])
            assert np.all(eigvals > 0), f"Class {i+1} covariance not positive-definite"

    def test_lut_dir_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_lut("/nonexistent/path")


# ── Normalisation ─────────────────────────────────────────────────────────────
class TestNormaliseRrs:
    def test_output_shape_1d(self):
        rrs = np.array([[0.01, 0.012, 0.015, 0.013, 0.008, 0.002]])
        Y = normalise_rrs(rrs)
        assert Y.shape == (1, 6)

    def test_output_shape_batch(self):
        rrs = np.random.exponential(0.005, (50, 6))
        Y = normalise_rrs(rrs)
        assert Y.shape == (50, 6)

    def test_nan_propagation(self):
        """NaN in any band should propagate to the whole row."""
        rrs = np.array([[np.nan, 0.01, 0.012, 0.01, 0.006, 0.001]])
        Y = normalise_rrs(rrs)
        assert np.all(np.isnan(Y))

    def test_nonpositive_masked(self):
        """Zero or negative Rrs values should produce NaN."""
        rrs = np.array([[0.0, 0.01, 0.012, 0.01, 0.006, 0.001]])
        Y = normalise_rrs(rrs)
        assert np.all(np.isnan(Y))

    def test_normalisation_invariance(self):
        """Scaling Rrs by a constant should not change Y (because we normalise by area)."""
        rrs = np.array([[0.01, 0.012, 0.015, 0.013, 0.008, 0.002]])
        Y1 = normalise_rrs(rrs)
        Y2 = normalise_rrs(rrs * 10)
        np.testing.assert_allclose(Y1, Y2, atol=1e-10)


# ── Classification ────────────────────────────────────────────────────────────
class TestClassify:
    def test_output_range(self, lut):
        """All valid classifications must be in range [1, 17]."""
        mu, sigma = lut
        rrs = np.random.exponential(0.005, (100, 6))
        classes = classify(rrs, mu, sigma)
        valid = classes[~np.isnan(classes)]
        assert np.all(valid >= 1) and np.all(valid <= 17)

    def test_nan_input_gives_nan(self, lut):
        mu, sigma = lut
        rrs = np.array([[np.nan] * 6])
        classes = classify(rrs, mu, sigma)
        assert np.isnan(classes[0])

    def test_deterministic(self, lut):
        """Same input must always give same output."""
        mu, sigma = lut
        rrs = np.array([[0.004, 0.005, 0.006, 0.005, 0.003, 0.001]])
        c1 = classify(rrs, mu, sigma)
        c2 = classify(rrs, mu, sigma)
        np.testing.assert_array_equal(c1, c2)

    def test_mixed_valid_invalid(self, lut):
        mu, sigma = lut
        rrs = np.array([
            [0.004, 0.005, 0.006, 0.005, 0.003, 0.001],  # valid
            [np.nan, 0.005, 0.006, 0.005, 0.003, 0.001], # invalid
            [0.010, 0.008, 0.007, 0.006, 0.005, 0.002],  # valid
        ])
        classes = classify(rrs, mu, sigma)
        assert not np.isnan(classes[0])
        assert np.isnan(classes[1])
        assert not np.isnan(classes[2])

    def test_known_coastal_spectrum(self, lut):
        """A turbid coastal spectrum should map to classes 1-9 (ANNa)."""
        mu, sigma = lut
        # High Rrs, relatively flat spectrum — typical turbid coastal water
        rrs = np.array([[0.025, 0.022, 0.020, 0.018, 0.015, 0.008]])
        c = classify(rrs, mu, sigma)[0]
        assert 1 <= c <= 9, f"Turbid coastal spectrum mapped to class {c}, expected 1-9"

    def test_2d_image(self, lut):
        mu, sigma = lut
        nlat, nlon = 10, 20
        rrs_bands = [np.random.exponential(0.005, (nlat, nlon)) for _ in range(6)]
        owc = classify_image(rrs_bands, LUT_DIR)
        assert owc.shape == (nlat, nlon)
        valid = owc[~np.isnan(owc)]
        assert np.all(valid >= 1) and np.all(valid <= 17)


# ── Model routing ─────────────────────────────────────────────────────────────
class TestOwcToDocModel:
    def test_coastal_classes(self):
        owc = np.array([1, 3, 5, 7, 9])
        result = owc_to_doc_model(owc)
        assert all(r == "ANNa" for r in result)

    def test_ocean_classes(self):
        owc = np.array([10, 12, 15, 17])
        result = owc_to_doc_model(owc)
        assert all(r == "ANNb" for r in result)

    def test_nan_class(self):
        owc = np.array([np.nan])
        result = owc_to_doc_model(owc)
        assert result[0] == ""

    def test_mixed(self):
        owc = np.array([1, 10, np.nan, 9, 17])
        result = owc_to_doc_model(owc)
        assert result[0] == "ANNa"
        assert result[1] == "ANNb"
        assert result[2] == ""
        assert result[3] == "ANNa"
        assert result[4] == "ANNb"


# ── Integration: classify matchup data ───────────────────────────────────────
class TestMatchupClassification:
    def test_matchup_classifies(self, lut):
        """Should classify at least some of the matchup Rrs without errors."""
        import pandas as pd
        mu, sigma = lut
        matchup_path = os.path.join(os.path.dirname(__file__), "..", "data", "matchup.csv")
        if not os.path.exists(matchup_path):
            pytest.skip("matchup.csv not found")

        df = pd.read_csv(matchup_path)
        rrs_cols = ["Rrs_412_2","Rrs_443_2","Rrs_490_2","Rrs_510_2","Rrs_560_2","Rrs_670_2"]
        rrs = df[rrs_cols].values
        classes = classify(rrs, mu, sigma)

        n_valid = int(np.sum(~np.isnan(classes)))
        assert n_valid > 500, f"Expected >500 valid classifications, got {n_valid}"

        valid = classes[~np.isnan(classes)]
        assert np.all(valid >= 1) and np.all(valid <= 17)

        n_anna = int(np.sum((valid >= 1) & (valid <= 9)))
        n_annb = int(np.sum((valid >= 10) & (valid <= 17)))
        assert n_anna > 0, "Expected some ANNa (coastal) classifications"
        assert n_annb > 0, "Expected some ANNb (open ocean) classifications"
        print(f"\n  Classified {n_valid} matchup points: ANNa={n_anna}, ANNb={n_annb}")
