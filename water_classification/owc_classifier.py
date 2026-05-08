"""
owc_classifier.py
-----------------
Optical Water Class (OWC) classifier for the DOC-ANNs pipeline.

Translates the original MATLAB algorithm (OWC_4gabi.m, Mélin & Vantrepotte 2015)
to Python. Classifies each pixel into one of 17 optical water classes using the
Mahalanobis distance in the log-normalised Rrs(λ) space.

Classification logic
--------------------
For each pixel with valid Rrs at 6 bands [412, 443, 490, 510, 560, 670] nm:

  1. Normalise the spectrum:
       area = trapezoid integral of Rrs over the 6 bands
       Y    = log10(Rrs / area)   [6-element vector]

  2. Compute the Mahalanobis distance from Y to each of the 17 class centroids
     using the per-class covariance matrix:
       d_i = mahalanobis(Y, mu_i, Sigma_i)

  3. Assign the class with minimum distance:
       class = argmin(d_i) + 1    [1-based, matching MATLAB convention]

DOC-ANNs switch
---------------
  Classes  1 – 9  → DOC-ANNa  (optically complex / coastal waters)
  Classes 10 – 17 → DOC-ANNb  (clear open ocean waters)

Reference
---------
Mélin, F., & Vantrepotte, V. (2015). How optically diverse is the coastal ocean?
Remote Sensing of Environment, 160, 235-251.
https://doi.org/10.1016/j.rse.2015.01.023

Bonelli, A.G., et al. (2022). Remote Sensing of Environment, 281, 113227.
https://doi.org/10.1016/j.rse.2022.113227
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Band wavelengths used for normalisation (must match the LUT)
# ---------------------------------------------------------------------------
WAVELENGTHS = np.array([412.0, 443.0, 490.0, 510.0, 560.0, 670.0])
N_CLASSES   = 17
N_BANDS     = 6

# OWC threshold for DOC-ANNs model switch
COASTAL_CLASSES    = set(range(1, 10))   # classes 1–9  → DOC-ANNa
OPEN_OCEAN_CLASSES = set(range(10, 18))  # classes 10–17 → DOC-ANNb


def load_lut(lut_dir: str):
    """
    Load the look-up tables (mean vectors and covariance matrices) for all
    17 optical water classes.

    Parameters
    ----------
    lut_dir : str
        Path to the directory containing the LUT text files:
        ``mu_Stat16ClassCoastal_XX.txt`` and ``CovMat_Stat16ClassCoastal_XX.txt``

    Returns
    -------
    mu : ndarray, shape (17, 6)
        Class mean vectors in log-normalised Rrs space.
    sigma : ndarray, shape (17, 6, 6)
        Per-class covariance matrices.
    """
    mu    = np.zeros((N_CLASSES, N_BANDS))
    sigma = np.zeros((N_CLASSES, N_BANDS, N_BANDS))

    for i in range(1, N_CLASSES + 1):
        tag = f"{i:02d}"
        mu_path  = os.path.join(lut_dir, f"mu_Stat16ClassCoastal_{tag}.txt")
        cov_path = os.path.join(lut_dir, f"CovMat_Stat16ClassCoastal_{tag}.txt")

        if not os.path.exists(mu_path):
            raise FileNotFoundError(f"LUT file not found: {mu_path}")
        if not os.path.exists(cov_path):
            raise FileNotFoundError(f"LUT file not found: {cov_path}")

        mu[i - 1]    = np.loadtxt(mu_path)
        sigma[i - 1] = np.loadtxt(cov_path)

    return mu, sigma


def normalise_rrs(rrs: np.ndarray) -> np.ndarray:
    """
    Log-normalise Rrs spectra.

    For each spectrum, computes the area under the curve using the trapezoidal
    rule over the 6 bands, then returns log10(Rrs / area).

    Parameters
    ----------
    rrs : ndarray, shape (..., 6)
        Remote sensing reflectance [sr⁻¹] at bands [412, 443, 490, 510, 560, 670].
        Any shape is accepted as long as the last dimension is 6.

    Returns
    -------
    Y : ndarray, same shape as rrs
        Log-normalised spectra. NaN where rrs contains NaN or non-positive values.
    """
    wl = WAVELENGTHS
    rrs = np.asarray(rrs, dtype=float)

    # Mask non-positive values before log (negative Rrs = invalid)
    rrs_clean = np.where(rrs <= 0, np.nan, rrs)

    # Trapezoidal integration along the band axis (last axis)
    # area = sum over consecutive pairs: (wl[k+1]-wl[k]) * (rrs[k+1]+rrs[k]) / 2
    dw   = np.diff(wl)                              # (5,)
    mid  = (rrs_clean[..., 1:] + rrs_clean[..., :-1]) / 2.0   # (..., 5)
    area = np.sum(dw * mid, axis=-1, keepdims=True)            # (..., 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        Y = np.log10(rrs_clean / area)

    return Y


def classify(rrs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Assign each pixel to its optical water class (1–17).

    Parameters
    ----------
    rrs : ndarray, shape (N, 6)
        Rrs spectra for N pixels. Rows with any NaN are assigned class NaN.
    mu : ndarray, shape (17, 6)
        Class mean vectors from load_lut().
    sigma : ndarray, shape (17, 6, 6)
        Class covariance matrices from load_lut().

    Returns
    -------
    owc : ndarray, shape (N,), dtype float
        Optical water class (1–17) for each pixel, or NaN for invalid pixels.
    """
    rrs = np.atleast_2d(np.asarray(rrs, dtype=float))
    N   = rrs.shape[0]

    Y = normalise_rrs(rrs)   # (N, 6)

    # Pre-compute inverse covariance matrices once
    sigma_inv = np.zeros_like(sigma)
    for i in range(N_CLASSES):
        try:
            sigma_inv[i] = np.linalg.inv(sigma[i])
        except np.linalg.LinAlgError:
            sigma_inv[i] = np.linalg.pinv(sigma[i])

    # Mahalanobis distance: d² = (y - mu)ᵀ Σ⁻¹ (y - mu)
    dist = np.full((N, N_CLASSES), np.nan)
    valid_mask = ~np.any(np.isnan(Y), axis=1)   # pixels with complete spectra

    if valid_mask.any():
        Y_valid = Y[valid_mask]                  # (M, 6)
        for i in range(N_CLASSES):
            diff = Y_valid - mu[i]               # (M, 6)
            # d² = diag(diff @ Σ⁻¹ @ diffᵀ)
            d2 = np.einsum("mi,ij,mj->m", diff, sigma_inv[i], diff)
            dist[valid_mask, i] = np.sqrt(np.maximum(d2, 0.0))

    # Class = index of minimum distance (1-based, matching MATLAB)
    owc = np.full(N, np.nan)
    rows_with_dist = ~np.all(np.isnan(dist), axis=1)
    owc[rows_with_dist] = np.nanargmin(dist[rows_with_dist], axis=1) + 1

    return owc


def classify_image(rrs_bands,
                   lut_dir: str) -> np.ndarray:
    """
    Classify a full satellite image (2D spatial grid).

    Parameters
    ----------
    rrs_bands : list of 6 ndarray, each shape (nlat, nlon)
        Rrs at [412, 443, 490, 510, 560, 670] nm, in that order.
    lut_dir : str
        Path to the LUT directory (containing mu_* and CovMat_* files).

    Returns
    -------
    owc_map : ndarray, shape (nlat, nlon)
        OWC class map (1–17), NaN over land or invalid pixels.

    Example
    -------
    >>> from water_classification.owc_classifier import classify_image
    >>> owc = classify_image([rrs412, rrs443, rrs490, rrs510, rrs560, rrs670],
    ...                      lut_dir="models/OWC_LUT")
    """
    mu, sigma = load_lut(lut_dir)

    shape = rrs_bands[0].shape
    N = shape[0] * shape[1]

    # Stack and flatten: (N, 6)
    rrs_flat = np.stack([b.reshape(N) for b in rrs_bands], axis=1)

    owc_flat = classify(rrs_flat, mu, sigma)
    return owc_flat.reshape(shape)


def owc_to_doc_model(owc: np.ndarray) -> np.ndarray:
    """
    Map OWC class values to DOC model labels.

    Parameters
    ----------
    owc : ndarray
        Array of OWC classes (1–17, or NaN).

    Returns
    -------
    model_mask : ndarray of str, same shape as owc
        "ANNa" for classes 1–9, "ANNb" for classes 10–17, "" for NaN.
    """
    owc = np.asarray(owc, dtype=float)
    result = np.full(owc.shape, "", dtype=object)
    result[(owc >= 1)  & (owc <= 9)]  = "ANNa"
    result[(owc >= 10) & (owc <= 17)] = "ANNb"
    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    lut_dir = os.path.join(os.path.dirname(__file__), "..", "models", "OWC_LUT")
    if not os.path.isdir(lut_dir):
        print(f"LUT directory not found: {lut_dir}")
        sys.exit(1)

    print("Loading LUTs...")
    mu, sigma = load_lut(lut_dir)
    print(f"  mu shape:    {mu.shape}")
    print(f"  sigma shape: {sigma.shape}")

    # Synthetic test spectra: one coastal (high Rrs, turbid) and one clear ocean
    rrs_test = np.array([
        [0.010, 0.008, 0.007, 0.006, 0.005, 0.002],   # turbid coastal
        [0.004, 0.005, 0.006, 0.005, 0.003, 0.001],   # typical open ocean
        [0.001, 0.002, 0.003, 0.003, 0.002, 0.0005],  # oligotrophic
        [np.nan, 0.003, 0.004, 0.003, 0.002, 0.001],  # invalid (NaN)
    ])

    classes = classify(rrs_test, mu, sigma)
    models  = owc_to_doc_model(classes)

    print("\nTest classification:")
    labels = ["turbid coastal", "typical ocean", "oligotrophic", "invalid"]
    for lbl, c, m in zip(labels, classes, models):
        print(f"  {lbl:<18} → class {str(int(c)) if not np.isnan(c) else 'NaN':>3}  →  {m or 'N/A'}")
