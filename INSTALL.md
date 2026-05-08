# Installation guide

Step-by-step instructions for **Mac**, **Linux**, and **Windows**.

TensorFlow is installed separately because the package name differs by platform.

---

## Mac — Apple Silicon (M1 / M2 / M3)

```bash
# 1. Clone the repository
git clone https://github.com/agbonelli/DOC-ANNs.git
cd DOC-ANNs

# 2. Create the conda environment (all deps except TensorFlow)
conda env create -f environment.yml
conda activate doc-anns

# 3. Install TensorFlow for Apple Silicon
pip install tensorflow-macos

# 4. Verify — 56 tests should pass
pytest tests/ -v

# 5. Generate scalers (once after install)
python scripts/regenerate_scalers.py

# 6. Run the model
python run_DOCNNs.py
```

> **Optional:** install `tensorflow-metal` for GPU acceleration via the Metal API:
> ```bash
> pip install tensorflow-metal
> ```

---

## Mac — Intel

```bash
conda env create -f environment.yml
conda activate doc-anns
pip install tensorflow-cpu
pytest tests/ -v
python scripts/regenerate_scalers.py
python run_DOCNNs.py
```

---

## Linux (including HPC clusters)

```bash
conda env create -f environment.yml
conda activate doc-anns
pip install tensorflow-cpu
pytest tests/ -v
python scripts/regenerate_scalers.py
python run_DOCNNs.py
```

### HPC with SLURM

```bash
module load anaconda3
conda env create -f environment.yml
conda activate doc-anns
pip install tensorflow-cpu
python scripts/regenerate_scalers.py
```

Batch script `run_doc.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=doc-anns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source activate doc-anns
python run_DOCNNs.py
```

Submit: `sbatch run_doc.sh`

---

## Windows

Open **Anaconda Prompt**:

```bat
conda env create -f environment.yml
conda activate doc-anns
pip install tensorflow-cpu
pytest tests/ -v
python scripts/regenerate_scalers.py
python run_DOCNNs.py
```

> **Paths in `run_DOCNNs.py`:** use forward slashes or raw strings:
> ```python
> DIR_SST = "C:/Users/yourname/data/SST"
> # or:
> DIR_SST = r"C:\Users\yourname\data\SST"
> ```

---

## Add model weights

The `.h5` files are distributed separately — contact **abonelli@asu.edu**.

```bash
cp /path/to/DOCANNa.h5  models/
cp /path/to/DOCANNb.h5  models/
```

The `.pkl` scalers are generated automatically by `scripts/regenerate_scalers.py`
and are already compatible with your environment.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
TensorFlow was not installed. Run the pip command for your platform:
```bash
pip install tensorflow-macos    # Apple Silicon
pip install tensorflow-cpu      # Intel / Linux / Windows
```

**`ModuleNotFoundError: No module named 'numpy._core'`**
The `.pkl` scaler was created with a different numpy version. Regenerate:
```bash
python scripts/regenerate_scalers.py
```

**`ModuleNotFoundError: No module named 'water_classification'`**
Always run scripts from the root of the repository:
```bash
cd /path/to/DOC-ANNs
python run_DOCNNs.py
```

**TensorFlow CUDA / GPU warnings**
Harmless on CPU-only machines. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2    # Mac / Linux
set TF_CPP_MIN_LOG_LEVEL=2       # Windows
```

**`netCDF4` fails on Windows**
```bash
conda install -c conda-forge netCDF4
```

**Updating**
```bash
git pull
conda env update -f environment.yml --prune
pip install tensorflow-macos     # or tensorflow-cpu
python scripts/regenerate_scalers.py
```
