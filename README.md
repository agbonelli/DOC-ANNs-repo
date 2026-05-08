# DOC-ANNs: Dissolved Organic Carbon from satellite remote sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.rse.2022.113227-blue)](https://doi.org/10.1016/j.rse.2022.113227)
[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)]()

Artificial Neural Networks to estimate surface **Dissolved Organic Carbon (DOC)**
concentration from satellite remote sensing in the global ocean.

> **Bonelli, A.G., Loisel, H., Jorge, D.S.F., Mangin, A., Fanton d'Andon, O., & Vantrepotte, V. (2022).**
> *A new method to estimate the dissolved organic carbon concentration from remote sensing in the global open ocean.*
> Remote Sensing of Environment, 281, 113227.
> https://doi.org/10.1016/j.rse.2022.113227

---

## Overview

The **DOC-ANNs** system combines two neural networks routed by Optical Water Class (OWC):

| Model | Water type | OWC classes | Inputs |
|-------|-----------|-------------|--------|
| **DOC-ANNa** | Coastal / optically complex | 1 – 9 | SST, aCDOM(443), MLD, Chl-a |
| **DOC-ANNb** | Clear open ocean | 10 – 17 | SST, aCDOM(443), MLD |

All inputs are satellite-derived from [GlobColour](https://hermes.acri.fr/) at 25 km / 8-day resolution,
with temporal lags of 1–2 weeks relative to the DOC target date.

### Performance (Bonelli et al., 2022)

| Dataset | N | RMSD (µmol/L) | MAPD (%) | MB (µmol/L) |
|---------|---|---------------|----------|-------------|
| DOC-ANNa — Training | 109 | 5.83 | 6.71 | −0.27 |
| DOC-ANNa — Validation | 47 | 8.00 | 9.44 | 1.26 |
| DOC-ANNb — Training | 215 | 5.59 | 6.32 | −1.08 |
| DOC-ANNb — Validation | 93 | 6.16 | 7.25 | −0.08 |

---

## Repository structure

```
DOC-ANNs/
├── README.md
├── LICENSE
├── requirements.txt              ← pip install
├── environment.yml               ← conda install
├── run_DOCNNs.py                 ← main inference script (NetCDF maps)
│
├── water_classification/         ← OWC classifier module
│   ├── __init__.py
│   └── owc_classifier.py        ← Python port of Mélin & Vantrepotte (2015)
│
├── models/
│   ├── OWC_LUT/                 ← 34 LUT files (mu + covariance per class)
│   ├── DOCANNa.h5               ← pre-trained weights (see below)
│   ├── DOCANNa_scaler.pkl
│   ├── DOCANNb.h5
│   └── DOCANNb_scaler.pkl
│
├── data/
│   ├── matchup.csv              ← full in-situ matchup database (4346 stations)
│   ├── DOCANNa_Train.csv        ← training set ANNa (N=109)
│   ├── DOCANNa_Val.csv          ← validation set ANNa (N=47)
│   ├── DOCANNb_Train.csv        ← training set ANNb (N=215)
│   └── DOCANNb_Val.csv          ← validation set ANNb (N=93)
│
├── notebooks/
│   ├── 01_model_performance.ipynb   ← reproduce Figure 4 (scatter plots)
│   ├── 02_model_architecture.ipynb  ← reproduce Figure 5 (flowchart)
│   ├── 03_global_DOC_map.ipynb      ← generate a global DOC map
│   └── 04_water_classification.ipynb← OWC classifier demo
│
├── scripts/
│   └── regenerate_scalers.py    ← rebuild .pkl files from training data
│
├── examples/
│   ├── predict_from_csv.py      ← predict DOC from tabular data (CLI)
│   └── predict_from_netcdf.py   ← full pipeline from satellite NetCDF
│
├── tests/                       ← 56 unit tests (pytest)
│   ├── test_owc.py
│   ├── test_predict.py
│   └── test_filename_lag.py
│
├── figures/                     ← pre-generated output figures
└── satellite_data/              ← put your GlobColour NetCDF files here
```

> **Model weights:** The `.h5` and `.pkl` files are not included due to size.
> Contact **abonelli@asu.edu** to request them, or see `scripts/regenerate_scalers.py`
> to rebuild the scalers from the included training data.

---

## Installation

Requires **conda** (Miniconda or Anaconda).
Full instructions for Mac, Linux, Windows, and HPC clusters: **[INSTALL.md](INSTALL.md)**

**Quick start (Mac / Linux):**

```bash
git clone https://github.com/agbonelli/DOC-ANNs.git
cd DOC-ANNs
conda env create -f environment.yml
conda activate doc-anns
pytest tests/ -v                        # 56 tests should pass
python scripts/regenerate_scalers.py    # generate .pkl files (once)
```

> **Apple Silicon (M1/M2/M3):** after creating the environment, replace
> `tensorflow-cpu` with `tensorflow-macos` — see [INSTALL.md](INSTALL.md).

---

## Add model weights

The pre-trained `.h5` model files are distributed separately (contact abonelli@asu.edu).
Once you have them, copy them to `models/`:

```bash
cp /path/to/DOCANNa.h5  models/
cp /path/to/DOCANNb.h5  models/
```

Then regenerate the scalers (needed once, matches your Python environment):

```bash
python scripts/regenerate_scalers.py
```

---

## Run the model

### Produce global DOC maps from satellite NetCDF files

**1.** Edit the `CONFIGURATION` block at the top of `run_DOCNNs.py` to point to your data folders:

```python
DIR_SST  = "/path/to/your/SST"
DIR_CDOM = "/path/to/your/CDOM443BL1"
DIR_MLD  = "/path/to/your/MLD"
DIR_CHL  = "/path/to/your/CHL1"
DIR_RRS  = "/path/to/your/RRS"

OUTPUT_DIR = "/path/to/your/DOC_output"
```

**2.** Run:

```bash
python run_DOCNNs.py
```

Expected output:
```
Loading models and scalers...
  DOC-ANNa loaded
  DOC-ANNb loaded
Loading OWC LUTs...
  17 classes loaded

Scanning satellite data folders...
  SST         :  846 files  (1997-01-01 → 2018-12-27)
  CDOM443BL1  :  846 files  (1997-01-01 → 2018-12-27)
  ...

  1997-01-01  ANNa=  234,521px  ANNb=1,823,445px  valid=2,057,966  mean=63.4 µmol/L
  1997-01-09  ...
```

### Predict DOC from a CSV file

```bash
# Demo (uses the included validation data, no model files needed)
python examples/predict_from_csv.py --model ANNb

# With your own data
python examples/predict_from_csv.py \
    --model ANNb \
    --input my_data.csv \
    --output my_data_DOC.csv
```

### Open the interactive notebooks

```bash
jupyter notebook notebooks/

# Start with:
# 01_model_performance.ipynb  — reproduce Figure 4 (no model files needed)
# 04_water_classification.ipynb — explore the OWC classifier
```

---

## Satellite data sources

| Variable | Product | Portal |
|----------|---------|--------|
| Chl-a (OC4) | GlobColour L3m 8-day 25 km | https://hermes.acri.fr |
| SST | GlobColour / OSTIA | https://hermes.acri.fr |
| aCDOM(443) | GlobColour L3m — Bonelli et al. (2021) algorithm | [https://hermes.acri.fr](https://doi.org/10.1016/j.rse.2021.112637) |
| MLD | CMEMS global ocean physics | https://marine.copernicus.eu |
| Rrs(412–670) | GlobColour L3m | https://hermes.acri.fr |

**Filename convention:**

```
SST / CDOM / MLD / CHL:
  L3m_[YYYYMMDD]-[YYYYMMDD]__GLOB_25_GSM-[Sensor]_[VAR]_8D_00.nc

Rrs (in RrsComplete/USED-RRS_GC/):
  L3m_[YYYYMMDD]-[YYYYMMDD]__GLOB_25_[Sensor]_NRRS[412|443|490|510|560|670]_8D_00.nc
```

**Time lag convention:**
- SST, MLD, Chl-a, Rrs → same 8-day composite as DOC target date
- aCDOM(443) → one 8-day composite earlier (−8 days)

---

## Quick API reference

```python
# OWC classification
from water_classification import load_lut, classify, classify_image

mu, sigma = load_lut("models/OWC_LUT")
owc = classify(rrs_array, mu, sigma)          # rrs shape: (N, 6)
owc_map = classify_image(
    [rrs412, rrs443, rrs490, rrs510, rrs560, rrs670],
    lut_dir="models/OWC_LUT"
)

# DOC prediction
import tensorflow as tf, joblib

model  = tf.keras.models.load_model("models/DOCANNb.h5")
scaler = joblib.load("models/DOCANNb_scaler.pkl")
X      = scaler.transform(df[["SST_table_2","cdom443_BL1_3","MLD_table_2"]].dropna())
DOC    = model.predict(X).squeeze()           # µmol/L
```

---

## Run the tests

```bash
pytest tests/ -v
# 56 passed
```

---

## Citation

```bibtex
@article{Bonelli2022,
  title   = {A new method to estimate the dissolved organic carbon concentration
             from remote sensing in the global open ocean},
  author  = {Bonelli, Ana Gabriela and Loisel, Hubert and Jorge, Daniel S.F.
             and Mangin, Antoine and {Fanton d'Andon}, Odile and Vantrepotte, Vincent},
  journal = {Remote Sensing of Environment},
  volume  = {281},
  pages   = {113227},
  year    = {2022},
  doi     = {10.1016/j.rse.2022.113227}
}
```

If you use the OWC classifier, also cite:

```bibtex
@article{Melin2015,
  title   = {How optically diverse is the coastal ocean?},
  author  = {M{\'e}lin, Fr{\'e}d{\'e}ric and Vantrepotte, Vincent},
  journal = {Remote Sensing of Environment},
  volume  = {160},
  pages   = {235--251},
  year    = {2015},
  doi     = {10.1016/j.rse.2015.01.023}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Contact

**Ana Gabriela Bonelli** — abonelli@asu.edu
