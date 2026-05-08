# ─────────────────────────────────────────────────────────────────────────────
# DOC-ANNs — Dockerfile
#
# Uses python:3.10-slim as base — works natively on both Apple Silicon
# (linux/arm64) and Intel (linux/amd64) without platform warnings.
# TensorFlow CPU is installed via pip.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="Ana Gabriela Bonelli <abonelli@asu.edu>"
LABEL description="DOC-ANNs: Dissolved Organic Carbon estimation from satellite remote sensing"
LABEL reference="Bonelli et al. (2022) Remote Sensing of Environment 281, 113227"

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ── Working directory and project files ───────────────────────────────────
WORKDIR /app
COPY . /app/

# Pre-compile to catch import errors at build time
RUN python -c "from water_classification import load_lut, classify, classify_image; print('water_classification OK')"
RUN python -c "import numpy, pandas, sklearn, netCDF4, matplotlib, scipy; print('all deps OK')"

# ── Jupyter configuration ──────────────────────────────────────────────────
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'"            >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False"       >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''"                 >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''"              >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True"          >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
CMD ["jupyter", "notebook", "--notebook-dir=/app/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
