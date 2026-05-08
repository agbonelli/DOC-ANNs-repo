"""
tests/test_filename_lag.py
--------------------------
Unit tests for filename parsing and temporal lag logic in run_DOCNNs.py.
Run with: pytest tests/ -v
"""

import sys
import os
from datetime import datetime, timedelta
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_DOCNNs import parse_filename, find_lagged_file, available_dates


# ── Filename parsing ──────────────────────────────────────────────────────────
class TestParseFilename:

    # Standard variables (SST / CDOM / MLD / CHL)
    def test_sst(self):
        fname = "L3m_20100101-20100108__GLOB_25_GSM-MODVIR_SST_8D_00.nc"
        d_start, d_end, var = parse_filename(fname)
        assert d_start == datetime(2010, 1, 1)
        assert d_end   == datetime(2010, 1, 8)
        assert var     == "SST"

    def test_cdom(self):
        fname = "L3m_20091224-20091231__GLOB_25_GSM-MODVIR_CDOM443BL1_8D_00.nc"
        d_start, _, var = parse_filename(fname)
        assert d_start == datetime(2009, 12, 24)
        assert var     == "CDOM443BL1"

    def test_mld(self):
        fname = "L3m_20100101-20100108__GLOB_25_GSM-MODVIR_MLD_8D_00.nc"
        assert parse_filename(fname)[2] == "MLD"

    def test_chl(self):
        fname = "L3m_20100101-20100108__GLOB_25_GSM-MODVIR_CHL1_8D_00.nc"
        assert parse_filename(fname)[2] == "CHL1"

    # Rrs format (no GSM-, uses NRRS prefix)
    def test_rrs443(self):
        fname = "L3m_20100101-20100108__GLOB_25_MODVIR_NRRS443_8D_00.nc"
        d_start, _, var = parse_filename(fname)
        assert d_start == datetime(2010, 1, 1)
        assert var     == "NRRS443"

    def test_rrs412(self):
        fname = "L3m_20100101-20100108__GLOB_25_MODVIR_NRRS412_8D_00.nc"
        assert parse_filename(fname)[2] == "NRRS412"

    def test_rrs670(self):
        fname = "L3m_20100101-20100108__GLOB_25_SeaWiFS_NRRS670_8D_00.nc"
        assert parse_filename(fname)[2] == "NRRS670"

    def test_full_path(self):
        path = "/data/RrsComplete/USED-RRS_GC/L3m_20100101-20100108__GLOB_25_MODVIR_NRRS443_8D_00.nc"
        result = parse_filename(path)
        assert result is not None
        assert result[2] == "NRRS443"

    def test_different_sensors(self):
        """Sensor name in filename should not affect parsing."""
        for sensor in ["MODVIR", "SeaWiFS", "MERIS", "OLCI"]:
            fname = f"L3m_20100101-20100108__GLOB_25_GSM-{sensor}_SST_8D_00.nc"
            assert parse_filename(fname)[2] == "SST"
            fname_rrs = f"L3m_20100101-20100108__GLOB_25_{sensor}_NRRS443_8D_00.nc"
            assert parse_filename(fname_rrs)[2] == "NRRS443"

    def test_wrong_format_returns_none(self):
        assert parse_filename("some_other_file.nc")   is None
        assert parse_filename("L3m_SST.nc")           is None
        assert parse_filename("")                       is None
        assert parse_filename("L3m_20100101__SST.nc") is None


# ── Lag resolution ────────────────────────────────────────────────────────────
class TestFindLaggedFile:
    """
    Synthetic catalog covering 4 consecutive 8-day periods:
        2009-12-24   period 0
        2010-01-01   period 1  (typical DOC target date)
        2010-01-09   period 2
        2010-01-17   period 3
    """

    @pytest.fixture
    def catalogs(self):
        dates = [datetime(2009,12,24), datetime(2010,1,1),
                 datetime(2010,1,9),   datetime(2010,1,17)]
        cat = {}
        for var in ["SST","CDOM443BL1","MLD","CHL1",
                    "NRRS412","NRRS443","NRRS490","NRRS510","NRRS560","NRRS670"]:
            cat[var] = {d: f"/data/{var}/{d.strftime('%Y%m%d')}.nc" for d in dates}
        return cat

    def test_lag0_same_period(self, catalogs):
        f = find_lagged_file(catalogs, "SST", datetime(2010,1,1), lag_periods=0)
        assert "20100101" in f

    def test_lag1_cdom_previous_period(self, catalogs):
        """Core lag: target=2010-01-01, CDOM lag=1 → must return 2009-12-24."""
        f = find_lagged_file(catalogs, "CDOM443BL1", datetime(2010,1,1), lag_periods=1)
        assert "20091224" in f

    def test_lag0_mld(self, catalogs):
        f = find_lagged_file(catalogs, "MLD", datetime(2010,1,9), lag_periods=0)
        assert "20100109" in f

    def test_lag0_all_rrs(self, catalogs):
        target = datetime(2010,1,1)
        for wl in [412,443,490,510,560,670]:
            f = find_lagged_file(catalogs, f"NRRS{wl}", target, lag_periods=0)
            assert f is not None and "20100101" in f, f"NRRS{wl} lag=0 failed"

    def test_missing_returns_none(self, catalogs):
        """lag=2 → date not in catalog → should return None, not raise."""
        f = find_lagged_file(catalogs, "SST", datetime(2010,1,1), lag_periods=2)
        assert f is None

    def test_missing_variable_returns_none(self, catalogs):
        f = find_lagged_file(catalogs, "NONEXISTENT", datetime(2010,1,1), lag_periods=0)
        assert f is None

    def test_year_boundary_lag(self, catalogs):
        """Lag across year boundary: target=2010-01-01, lag=1 → 2009-12-24."""
        f = find_lagged_file(catalogs, "CDOM443BL1", datetime(2010,1,1), lag_periods=1)
        assert "20091224" in f


# ── Full Bonelli et al. (2022) lag scenario ───────────────────────────────────
class TestBonelli2022LagScenario:
    """
    Verify the exact lag convention from the paper for a representative
    target date (2010-01-01):

      SST    lag=0 → 2010-01-01  ✓
      MLD    lag=0 → 2010-01-01  ✓
      CHL1   lag=0 → 2010-01-01  ✓
      CDOM   lag=1 → 2009-12-24  ✓  (one 8-day period earlier)
      NRRSxx lag=0 → 2010-01-01  ✓
    """

    @pytest.fixture
    def catalogs(self):
        dates = [datetime(2009,12,24), datetime(2010,1,1), datetime(2010,1,9)]
        cat = {}
        for var in ["SST","CDOM443BL1","MLD","CHL1",
                    "NRRS412","NRRS443","NRRS490","NRRS510","NRRS560","NRRS670"]:
            cat[var] = {d: f"/data/{var}/{d.strftime('%Y%m%d')}.nc" for d in dates}
        return cat

    def test_all_lags(self, catalogs):
        target = datetime(2010,1,1)

        assert "20100101" in find_lagged_file(catalogs, "SST",       target, 0)
        assert "20091224" in find_lagged_file(catalogs, "CDOM443BL1",target, 1)
        assert "20100101" in find_lagged_file(catalogs, "MLD",       target, 0)
        assert "20100101" in find_lagged_file(catalogs, "CHL1",      target, 0)

        for wl in [412,443,490,510,560,670]:
            f = find_lagged_file(catalogs, f"NRRS{wl}", target, 0)
            assert f is not None and "20100101" in f

    def test_available_dates_sorted(self, catalogs):
        dates = available_dates(catalogs, "SST")
        assert dates == sorted(dates)
        assert len(dates) == 3
