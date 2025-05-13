"""Functions for calculating features for the Geology Forecast Challenge.

This module defines the core logic for transforming input time series into
various derived features. Operations are vectorized or use `np.apply_along_axis`
for improved performance over row-by-row Python loops.
"""

import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pywt  # For wavelet features

# from scipy.fft import fft # np.fft.rfft is used instead for real inputs
from loguru import logger

# tqdm could be used inside np.apply_along_axis if custom progress is needed,
# but generally not used directly when the main loop is removed.

from config import (
    ColabInspiredFeaturesConfigModel,
    ColumnConfig,
    DerivedFeaturesConfig,
    OriginalDerivedFeaturesConfigModel,
    WaveletConfigModel,
)
from constants import InputSignalType


# --- Public Helper Functions ---


def get_valid_points_for_series(
    series_row: np.ndarray, x_values_row: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Processes a single 1D array to extract valid (non-NaN, non-Inf) points.

    If `x_values_row` are provided and match dimensions, corresponding valid
    x_values are also returned.

    Args:
        series_row: A 1D NumPy array representing a single time series.
        x_values_row: Optional 1D NumPy array of x-coordinates for series_row.

    Returns:
        A tuple containing:
            - A 1D NumPy array of valid series points.
            - An optional 1D NumPy array of corresponding valid x-points,
              or None if x_values_row was not provided or unusable.

    Raises:
        TypeError: If `series_row` is not a 1D NumPy array, or if
                   `x_values_row` (when provided) is not a 1D NumPy array.
    """
    if not isinstance(series_row, np.ndarray) or series_row.ndim != 1:
        raise TypeError(
            f"series_row must be a 1D numpy.ndarray, got {type(series_row)}"
        )

    valid_mask = ~np.isnan(series_row) & ~np.isinf(series_row)
    valid_series = series_row[valid_mask]
    valid_x_processed: Optional[np.ndarray] = None

    if x_values_row is not None:
        if not isinstance(x_values_row, np.ndarray) or x_values_row.ndim != 1:
            raise TypeError(
                f"x_values_row must be a 1D numpy.ndarray or None, "
                f"got {type(x_values_row)}"
            )
        if len(x_values_row) == len(series_row):
            valid_x_processed = x_values_row[valid_mask]
        elif len(x_values_row) == len(valid_series):  # x_values already filtered
            valid_x_processed = x_values_row
        else:
            logger.trace(  # Log quietly if x_values are unusable for this row
                f"x_values_row length ({len(x_values_row)}) mismatches series_row "
                f"({len(series_row)}) and valid_series ({len(valid_series)}). "
                "Cannot reliably use these x_values_row."
            )
    return valid_series, valid_x_processed


def calculate_series_std_dev(
    series_row: np.ndarray, default_val: float = 0.01
) -> float:
    """Calculates standard deviation for a single 1D series, handling NaNs.

    Args:
        series_row: The 1D NumPy array for which to calculate std dev.
        default_val: Value to return if std dev cannot be computed (e.g., empty).

    Returns:
        The calculated standard deviation or the default value.
    """
    valid_pts, _ = get_valid_points_for_series(series_row)
    if valid_pts.size > 0:
        return float(np.std(valid_pts, ddof=0))  # ddof=0 matches default behavior
    return default_val


# --- Directly Vectorized Basic Statistics ---
def _vectorized_mean(
    series_batch: np.ndarray, window_slice: slice, default_val: float = 0.0
) -> np.ndarray:
    """Calculates mean along axis 1 for a windowed slice of a batch."""
    data_slice = series_batch[:, window_slice]
    with warnings.catch_warnings():  # Suppress mean of empty slice warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(data_slice, axis=1)
    return np.nan_to_num(means, nan=default_val)


def _vectorized_std_dev(
    series_batch: np.ndarray, window_slice: slice, default_val: float = 0.01
) -> np.ndarray:
    """Calculates std dev along axis 1 for a windowed slice of a batch."""
    data_slice = series_batch[:, window_slice]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stds = np.nanstd(data_slice, axis=1, ddof=0)
    return np.nan_to_num(stds, nan=default_val)


# --- Row-wise Wrappers for np.apply_along_axis ---
def _calculate_polyfit_coeff_row_wrapper(
    series_row_with_x: np.ndarray,
    num_series_pts: int,
    degree: int,
    coeff_idx: int,
    default_val: float,
) -> float:
    """Wrapper for np.polyfit for use with np.apply_along_axis."""
    series_pts = series_row_with_x[:num_series_pts]
    x_pts = series_row_with_x[num_series_pts:]
    valid_series, valid_x = get_valid_points_for_series(series_pts, x_pts)

    if not (
        valid_x is not None
        and len(valid_series) == len(valid_x)
        and len(valid_series) >= degree + 1
    ):
        return default_val
    try:
        with warnings.catch_warnings():
            # RankWarning inherits from UserWarning in NumPy
            warnings.simplefilter("ignore", category=UserWarning)
            coeffs = np.polyfit(valid_x, valid_series, degree)
    except (np.linalg.LinAlgError, ValueError, TypeError) as e_fit:
        logger.trace(f"Polyfit (deg {degree}) failed for row: {e_fit}")
        return default_val
    return float(coeffs[coeff_idx]) if 0 <= coeff_idx < len(coeffs) else default_val


def _calculate_rate_of_change_row_wrapper(
    series_row: np.ndarray, default_val: float, relative: bool
) -> float:
    """Wrapper for rate of change for use with np.apply_along_axis."""
    pts, _ = get_valid_points_for_series(series_row)
    if len(pts) < 2:
        return default_val
    abs_change = pts[-1] - pts[0]
    if not relative:
        return float(abs_change)
    first_val = pts[0]
    if abs(first_val) > 1e-9:  # Avoid division by zero
        return float(abs_change / first_val)
    # Handle case: first_val is zero or near-zero
    return 0.0 if abs(abs_change) < 1e-9 else float(np.sign(abs_change) * 100.0)


def _calculate_mean_ratio_row_wrapper(
    series_row: np.ndarray, idx_prior_end: int, idx_prior_start: int, default_val: float
) -> float:
    """Wrapper for mean ratio for use with np.apply_along_axis."""
    s_recent_pts, _ = get_valid_points_for_series(series_row[idx_prior_end:])
    s_prior_pts, _ = get_valid_points_for_series(
        series_row[idx_prior_start:idx_prior_end]
    )

    mean_recent = np.mean(s_recent_pts) if s_recent_pts.size > 0 else np.nan
    mean_prior = np.mean(s_prior_pts) if s_prior_pts.size > 0 else np.nan

    if np.isnan(mean_recent) or np.isnan(mean_prior):
        return default_val
    if abs(mean_prior) > 1e-9:
        return float(mean_recent / mean_prior)
    if abs(mean_recent) < 1e-9:  # Both near zero
        return 1.0
    return float(np.sign(mean_recent) * 100.0 * default_val)  # Large ratio


# --- Vectorized Orchestrators for Each Feature Type ---
def _extract_original_derived_features_vectorized(
    series_batch: np.ndarray,
    x_values_full_input: np.ndarray,  # Shared 1D array or per-sample 2D array
    cfg: OriginalDerivedFeaturesConfigModel,
) -> np.ndarray:
    """Extracts 'original' derived features using vectorized/apply_along_axis."""
    logger.debug("Extracting original derived features (vectorized approach)...")
    n_samples, n_input_features = series_batch.shape
    features_batch = np.full((n_samples, cfg.n_features), np.nan, dtype=float)

    # Prepare x_values for the 'recent' window
    recent_win_slice = slice(-cfg.recent_points_window, None)
    series_recent_batch = series_batch[:, recent_win_slice]
    num_pts_recent_win = series_recent_batch.shape[1]

    x_recent_batch_data: np.ndarray
    if x_values_full_input.ndim == 1:
        x_recent_template = x_values_full_input[recent_win_slice]
        if len(x_recent_template) != num_pts_recent_win:
            raise ValueError(
                "Sliced shared x_values for recent window length mismatch."
            )
        x_recent_batch_data = np.tile(x_recent_template, (n_samples, 1))
    elif x_values_full_input.ndim == 2:
        if (
            x_values_full_input.shape[0] != n_samples
            or x_values_full_input.shape[1] != n_input_features
        ):
            raise ValueError("Per-sample x_values shape mismatch.")
        x_recent_batch_data = x_values_full_input[:, recent_win_slice]
    else:
        raise ValueError("x_values_full_input has invalid dimensions.")

    series_and_x_recent = np.concatenate(
        (series_recent_batch, x_recent_batch_data), axis=1
    )

    features_batch[:, 0] = np.apply_along_axis(
        _calculate_polyfit_coeff_row_wrapper,
        1,
        series_and_x_recent,
        num_series_pts=num_pts_recent_win,
        degree=1,
        coeff_idx=0,
        default_val=0.0,
    )
    features_batch[:, 1] = _vectorized_mean(
        series_batch, slice(-cfg.mvg_avg_window, None)
    )
    features_batch[:, 2] = _vectorized_std_dev(
        series_batch, slice(-cfg.std_dev_window, None)
    )
    features_batch[:, 3] = np.apply_along_axis(
        _calculate_polyfit_coeff_row_wrapper,
        1,
        series_and_x_recent,
        num_series_pts=num_pts_recent_win,
        degree=2,
        coeff_idx=0,
        default_val=0.0,
    )
    series_roc_batch = series_batch[:, slice(-cfg.roc_window, None)]
    features_batch[:, 4] = np.apply_along_axis(
        _calculate_rate_of_change_row_wrapper,
        1,
        series_roc_batch,
        default_val=0.0,
        relative=True,
    )
    features_batch[:, 5] = np.apply_along_axis(
        _calculate_mean_ratio_row_wrapper,
        1,
        series_batch,
        idx_prior_end=cfg.prior_window_end,
        idx_prior_start=cfg.prior_window_start,
        default_val=1.0,
    )

    return np.nan_to_num(features_batch, nan=0.0)


def _calculate_wavelet_features_single_row(
    series_row: np.ndarray, cfg: WaveletConfigModel
) -> np.ndarray:
    """Internal helper: calculates wavelet features for one series row."""
    min_len_for_decomp = pywt.Wavelet(cfg.family).dec_len
    actual_decomp_level = cfg.decomposition_level

    if cfg.decomposition_level > 0:
        if len(series_row) < min_len_for_decomp:
            actual_decomp_level = 0
        else:
            max_possible_level = pywt.dwt_max_level(
                len(series_row), pywt.Wavelet(cfg.family)
            )
            actual_decomp_level = min(
                max(0, max_possible_level), cfg.decomposition_level
            )
    else:
        actual_decomp_level = 0

    output_features = np.zeros(
        cfg.n_derived_features
    )  # Ensures correct shape with padding
    if actual_decomp_level < 1:
        return output_features

    try:
        coeffs = pywt.wavedec(
            series_row, cfg.family, level=actual_decomp_level, mode=cfg.mode
        )
        calculated_feature_list: List[float] = []

        for i in range(1, actual_decomp_level + 1):  # Detail coefficients
            detail_coeffs_level_i = coeffs[i]
            valid_details, _ = get_valid_points_for_series(detail_coeffs_level_i)
            calculated_feature_list.append(
                np.std(valid_details) if valid_details.size > 0 else 0.0
            )
            calculated_feature_list.append(
                np.max(np.abs(valid_details)) if valid_details.size > 0 else 0.0
            )

        approx_coeffs_level_actual = coeffs[0]  # Approximation coefficients
        valid_approx, _ = get_valid_points_for_series(approx_coeffs_level_actual)
        calculated_feature_list.append(
            np.std(valid_approx) if valid_approx.size > 0 else 0.0
        )
        calculated_feature_list.append(
            np.mean(valid_approx) if valid_approx.size > 0 else 0.0
        )

        # Place calculated features into the zero-padded output_features array
        num_detail_feats_calculated = actual_decomp_level * cfg.n_features_per_level
        output_features[:num_detail_feats_calculated] = calculated_feature_list[
            :num_detail_feats_calculated
        ]
        output_features[-cfg.n_features_final_approx :] = calculated_feature_list[
            num_detail_feats_calculated:
        ]
    except Exception as e_wav:
        logger.trace(f"Wavelet decomposition for a row failed: {e_wav}.")
        # output_features remains zeros
    return output_features


def _calculate_wavelet_features_vectorized(
    series_batch: np.ndarray, cfg: WaveletConfigModel
) -> np.ndarray:
    """Calculates wavelet features for a batch of series using apply_along_axis."""
    logger.debug("Calculating wavelet features (using apply_along_axis)...")
    if series_batch.shape[0] == 0:  # Handle empty batch
        return np.empty((0, cfg.n_derived_features))

    # This lambda is necessary because apply_along_axis needs a function
    # that takes only the 1D array slice, but our row processor needs 'cfg'.
    row_processor = lambda row_data: _calculate_wavelet_features_single_row(
        row_data, cfg
    )

    wavelet_features_batch = np.apply_along_axis(
        row_processor, axis=1, arr=series_batch
    )
    return np.nan_to_num(wavelet_features_batch, nan=0.0)


def _calculate_trend_strength_row_wrapper(
    series_row_with_x: np.ndarray, num_series_pts: int, trend_window_val: int
) -> float:
    """Calculates R-squared for trend line in a window for a single row."""
    series_segment = series_row_with_x[:num_series_pts][-trend_window_val:]
    x_segment = series_row_with_x[num_series_pts:][-trend_window_val:]

    valid_pts, valid_x = get_valid_points_for_series(series_segment, x_segment)
    if not (
        valid_x is not None and valid_pts.size == valid_x.size and valid_pts.size >= 2
    ):
        return 0.0  # Cannot calculate R-squared with fewer than 2 points
    if np.all(valid_pts == valid_pts[0]):  # Constant series
        return 1.0  # Perfect fit if constant (R^2 definition dependent, but typically 1 or undefined)
    try:
        slope, intercept = np.polyfit(valid_x, valid_pts, 1)
        predictions = slope * valid_x + intercept
        ss_residuals = np.sum((valid_pts - predictions) ** 2)
        ss_total = np.sum((valid_pts - np.mean(valid_pts)) ** 2)
        # R-squared: 1 - (SS_res / SS_tot). Clip to [0, 1] as poor fits can be negative.
        return float(np.clip(1.0 - (ss_residuals / (ss_total + 1e-9)), 0.0, 1.0))
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return 0.0


def _calculate_pattern_change_row_wrapper(
    series_row: np.ndarray, pattern_window_val: int
) -> float:
    """Compares mean/std of two halves of a window for a single row."""
    windowed_pts = series_row[-pattern_window_val:]
    valid_pts, _ = get_valid_points_for_series(windowed_pts)
    if valid_pts.size < 4:
        return 0.0  # Need at least 2 points per half

    mid_idx = valid_pts.size // 2
    first_half = valid_pts[:mid_idx]
    second_half = valid_pts[mid_idx:]

    if not (first_half.size > 0 and second_half.size > 0):
        return 0.0

    mean1 = np.mean(first_half)
    std1 = np.std(first_half, ddof=0) if first_half.size > 0 else 0.0
    std1 = (
        std1 if std1 > 1e-9 else 1e-9
    )  # Avoid division by zero for relative std change

    mean2 = np.mean(second_half)
    std2 = np.std(second_half, ddof=0) if second_half.size > 0 else 0.0
    # std2 is not used as a denominator here, so less critical for flooring

    # np.nanmean/np.nanstd might be safer if internal NaNs are possible AFTER get_valid_points
    rel_mean_change = abs(mean2 - mean1) / (abs(mean1) + 1e-6)
    rel_std_change = abs(std2 - std1) / (std1 + 1e-6)  # std1 floored
    return float((rel_mean_change + rel_std_change) / 2.0)


def _calculate_spectral_balance_row_wrapper(series_row: np.ndarray) -> float:
    """Calculates spectral balance for a single row."""
    valid_series_fft, _ = get_valid_points_for_series(series_row)
    if valid_series_fft.size < 10:
        return 0.0  # Need enough points for meaningful FFT

    windowed_fft_data = valid_series_fft * np.hanning(len(valid_series_fft))
    # Use rfft for real input; magnitudes are for positive frequencies
    fft_magnitudes = np.abs(np.fft.rfft(windowed_fft_data))[
        1:
    ]  # Exclude DC component (zeroth freq)

    if fft_magnitudes.size < 3:
        return 0.0  # Need at least 3 points for L/M/H bands

    num_fft_points_half = fft_magnitudes.size
    idx_low_end = num_fft_points_half // 3
    idx_mid_end = 2 * num_fft_points_half // 3

    power_low = np.sum(fft_magnitudes[:idx_low_end])
    power_mid = np.sum(fft_magnitudes[idx_low_end:idx_mid_end])
    power_high = np.sum(fft_magnitudes[idx_mid_end:])
    power_total = power_low + power_mid + power_high

    return (
        float((power_high - power_low) / (power_total + 1e-9))
        if power_total > 1e-9
        else 0.0
    )


def _extract_colab_inspired_features_vectorized(
    series_batch: np.ndarray,
    x_values_full_input: np.ndarray,
    cfg: ColabInspiredFeaturesConfigModel,
) -> np.ndarray:
    """Extracts Colab-inspired features using apply_along_axis."""
    logger.debug("Extracting Colab-inspired features (vectorized approach)...")
    n_samples, n_input_features = series_batch.shape
    features_batch = np.full((n_samples, cfg.n_features), np.nan, dtype=float)

    # Prepare concatenated data for trend strength
    series_and_x_data_for_trend: np.ndarray
    if x_values_full_input.ndim == 1:  # Shared x_values
        x_full_tiled = np.tile(x_values_full_input, (n_samples, 1))
        series_and_x_data_for_trend = np.concatenate(
            (series_batch, x_full_tiled), axis=1
        )
    else:  # Per-sample x_values
        series_and_x_data_for_trend = np.concatenate(
            (series_batch, x_values_full_input), axis=1
        )

    features_batch[:, 0] = np.apply_along_axis(
        _calculate_trend_strength_row_wrapper,
        1,
        series_and_x_data_for_trend,
        num_series_pts=n_input_features,
        trend_window_val=cfg.trend_strength_window,
    )
    features_batch[:, 1] = np.apply_along_axis(
        _calculate_pattern_change_row_wrapper,
        1,
        series_batch,
        pattern_window_val=cfg.pattern_change_window_split,
    )
    features_batch[:, 2] = np.apply_along_axis(
        _calculate_spectral_balance_row_wrapper, 1, series_batch
    )

    return np.nan_to_num(features_batch, nan=0.0)


# --- Main Orchestration for Derived Features (VECTORIZED) ---
def create_all_derived_features_vectorized(
    X_imputed: np.ndarray,
    derived_feat_cfg: DerivedFeaturesConfig,
    col_cfg: ColumnConfig,
) -> np.ndarray:
    """Orchestrates vectorized creation of all derived features."""
    n_samples, n_input_features = X_imputed.shape
    if n_input_features != col_cfg.input_length:
        logger.warning(
            f"Input feature count {n_input_features} in vectorized FE "
            f"mismatches config {col_cfg.input_length}."
        )
    if derived_feat_cfg.colab is None:
        raise ValueError("DerivedFeaturesConfig.colab attribute not initialized.")

    # x_values_input_domain assumes 0-based indexing matching array columns
    # The col_cfg.input_col_range_start/end refers to column *names*.
    # For actual x-coordinates if features need them, pass actual coordinates or relative indices.
    # For now, assuming x_vals_input_domain can be a simple range for features like polyfit.
    x_vals_input_domain = np.arange(n_input_features)  # Simple 0 to M-1 indices

    # If actual world coordinates are needed, this would come from elsewhere or be computed
    # e.g. x_vals_world_coords = np.arange(col_cfg.input_col_range_start, col_cfg.input_col_range_end + 1)
    # and then ensured it aligns with X_imputed's columns if passed.
    # For simplicity with current structure, using simple indices [0, M-1] as default x_values.

    logger.info("Starting vectorized extraction of original derived features...")
    orig_feats = _extract_original_derived_features_vectorized(
        X_imputed, x_vals_input_domain, derived_feat_cfg.original
    )
    logger.info("Starting vectorized extraction of wavelet features...")
    wavelet_feats = _calculate_wavelet_features_vectorized(
        X_imputed, derived_feat_cfg.wavelet
    )
    logger.info("Starting vectorized extraction of Colab-inspired features...")
    colab_feats = _extract_colab_inspired_features_vectorized(
        X_imputed, x_vals_input_domain, derived_feat_cfg.colab
    )

    all_derived_feats = np.concatenate((orig_feats, wavelet_feats, colab_feats), axis=1)

    if np.isnan(all_derived_feats).any() or np.isinf(all_derived_feats).any():
        logger.warning(
            "NaNs/Infs found in final derived features. Applying nan_to_num."
        )
        all_derived_feats = np.nan_to_num(
            all_derived_feats, nan=0.0, posinf=0.0, neginf=0.0
        )

    expected_cols = derived_feat_cfg.total_n_derived_features
    if all_derived_feats.shape[1] != expected_cols:
        logger.critical(
            f"CRITICAL: Vectorized features dim mismatch! Exp {expected_cols}, "
            f"Got {all_derived_feats.shape[1]}. Padding/truncating to recover."
        )
        actual_cols = all_derived_feats.shape[1]
        if actual_cols > expected_cols:
            all_derived_feats = all_derived_feats[:, :expected_cols]
        else:
            padding = np.zeros((n_samples, expected_cols - actual_cols))
            all_derived_feats = np.hstack((all_derived_feats, padding))
    return all_derived_feats


# Alias for external calls if `create_all_derived_features` name is expected
create_all_derived_features = create_all_derived_features_vectorized


# --- Signal Characterization ---
def characterize_input_signal(
    input_series: np.ndarray,
    wavelet_cfg: WaveletConfigModel,
    realiz_calib_cfg: "config.RealizationCalibrationConfig",
) -> InputSignalType:
    """Characterizes input signal complexity using wavelet decomposition."""
    min_len_any_decomp = pywt.Wavelet(wavelet_cfg.family).dec_len
    actual_level = wavelet_cfg.decomposition_level

    if actual_level > 0:  # Check if decomposition is even requested
        if len(input_series) < min_len_any_decomp:
            actual_level = 0  # Not enough data for even one level
        else:
            max_possible_level = pywt.dwt_max_level(
                len(input_series), pywt.Wavelet(wavelet_cfg.family)
            )
            # Use requested level, but cap it by what's possible and ensure it's non-negative
            actual_level = min(
                max(0, max_possible_level), wavelet_cfg.decomposition_level
            )
    else:  # Requested decomposition level is 0 or less
        actual_level = 0

    if actual_level < 1:
        return InputSignalType.COMPLEX  # Default if no useful decomposition

    try:
        coeffs = pywt.wavedec(
            input_series, wavelet_cfg.family, level=actual_level, mode=wavelet_cfg.mode
        )

        # Detail coeffs (cD_L, ..., cD_1) are coeffs[1] through coeffs[actual_level+1-1]
        detail_coeffs_list = [c for c in coeffs[1:] if c is not None and c.size > 0]
        all_detail_coeffs = (
            np.concatenate(detail_coeffs_list) if detail_coeffs_list else np.array([])
        )

        approx_coeffs = coeffs[0] if coeffs[0] is not None else np.array([])

        energy_approx = np.sum(approx_coeffs**2)
        energy_detail = np.sum(all_detail_coeffs**2)
        total_energy = energy_approx + energy_detail

        if total_energy < 1e-9:
            return InputSignalType.SMOOTH  # No energy, likely flat or zero

        detail_ratio = energy_detail / total_energy
        max_abs_detail_coeff = (
            np.max(np.abs(all_detail_coeffs)) if all_detail_coeffs.size > 0 else 0.0
        )

        if max_abs_detail_coeff > realiz_calib_cfg.input_faulty_thresh_max_detail:
            return InputSignalType.POTENTIALLY_DISCONTINUOUS
        if detail_ratio < realiz_calib_cfg.input_smooth_thresh_ratio:
            return InputSignalType.SMOOTH
        return InputSignalType.COMPLEX

    except Exception as e_char:
        logger.warning(
            f"Failed to characterize input signal: {e_char}. Defaulting to COMPLEX."
        )
        return InputSignalType.COMPLEX
