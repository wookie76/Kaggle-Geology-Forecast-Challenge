"""Handles model evaluation, NLL calculation, calibration, and variance analysis.

This module provides functions for:
- Calculating the Negative Log Likelihood (NLL) loss.
- Calibrating the NLL variance scale factor for heuristic realizations.
- Analyzing the variance structure of training data to inform realization
  generation.
- Computing the idealized inverse covariance vector for NLL calculation.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from config import AppConfig, ColumnConfig, RealizationCalibrationConfig
from constants import CovarianceRegion

# Corrected import: get_valid_points_for_series is now public
from feature_engineering import get_valid_points_for_series

# The following import assumes generate_heuristic_realizations is now the
# public function in realization_generation.py for generating unscaled realizations.
# If it was named generate_calibrated_realizations there, that name should be used.
# For consistency with the NLL calibration loop, this is expected.
import realization_generation  # For generate_heuristic_realizations

# Using the type alias for readability
OptimizedModelParams = Dict[str, Dict[str, Any]]


@lru_cache(maxsize=None)  # Cache results to avoid recomputation
def get_inverse_covariance_vector(
    n_outputs: int,
    log_slopes_tuple: Tuple[float, ...],
    log_offsets_tuple: Tuple[float, ...],
) -> np.ndarray:
    """Computes the idealized inverse covariance vector for NLL.

    This vector, D_T^{-1}(x) from the competition description, is derived
    from curve-fitting based on training data characteristics. It has three
    log-linear segments.

    Args:
        n_outputs: The number of output positions (e.g., 300).
        log_slopes_tuple: A tuple of three slope values (ak) for the
                          log-linear segments.
        log_offsets_tuple: A tuple of three offset values (bk) for the
                           log-linear segments.

    Returns:
        A 1D NumPy array of inverse covariance values, one for each output position.

    Raises:
        ValueError: If log_slopes_tuple or log_offsets_tuple do not contain
                    exactly three elements.
    """
    if len(log_slopes_tuple) != 3 or len(log_offsets_tuple) != 3:
        raise ValueError(
            "log_slopes_tuple and log_offsets_tuple must each have 3 elements, "
            "corresponding to the three covariance regions."
        )

    inv_covariance_vector = np.zeros(n_outputs)
    for x_pos_1_based in range(1, n_outputs + 1):  # Iterate 1 to n_outputs
        region_idx_val: int
        if 1 <= x_pos_1_based <= 60:  # Region 1
            region_idx_val = CovarianceRegion.EXPONENTIAL_GROWTH.value
        elif 61 <= x_pos_1_based <= 244:  # Region 2
            region_idx_val = CovarianceRegion.STABLE_ERRORS.value
        else:  # Region 3 (245 to n_outputs)
            region_idx_val = CovarianceRegion.FURTHER_GROWTH.value

        log_x = np.log(x_pos_1_based)
        # Value is exp(log(x)*ak + bk)
        val = np.exp(
            log_x * log_slopes_tuple[region_idx_val] + log_offsets_tuple[region_idx_val]
        )
        inv_covariance_vector[x_pos_1_based - 1] = val  # Store in 0-indexed array

    return inv_covariance_vector


def calculate_nll_loss(
    y_true: np.ndarray, pred_reals: np.ndarray, app_config: AppConfig
) -> float:
    """Calculates the Negative Log Likelihood (NLL) loss.

    Args:
        y_true: True target values, shape (n_samples, n_outputs).
        pred_reals: Predicted realizations, shape
                    (n_samples, n_realizations, n_outputs).
        app_config: Application configuration object.

    Returns:
        The mean NLL loss over all samples.

    Raises:
        ValueError: If shapes of y_true and pred_reals are incompatible,
                    or if inverse covariance vector length mismatches.
    """
    n_s, n_r, n_o = pred_reals.shape
    if y_true.shape != (n_s, n_o):
        raise ValueError(
            f"Shape mismatch for NLL: y_true={y_true.shape}, "
            f"pred_reals={pred_reals.shape}"
        )

    inv_cov_vector = get_inverse_covariance_vector(
        n_outputs=n_o,
        log_slopes_tuple=app_config.realization_calib.log_slopes,
        log_offsets_tuple=app_config.realization_calib.log_offsets,
    )

    if len(inv_cov_vector) != n_o:
        raise ValueError(
            f"InvCov vector length {len(inv_cov_vector)} != n_outputs {n_o}."
        )

    p_i = 1.0 / n_r if n_r > 0 else 1.0  # Probability per kernel

    sample_losses = np.zeros(n_s)
    for i in range(n_s):  # Iterate over samples
        sum_exp_misfits_for_sample = 0.0
        for r_j in range(n_r):  # Iterate over realizations for this sample
            error_vector = y_true[i] - pred_reals[i, r_j, :]
            # Misfit term: sum(error_vector^2 * inv_cov_vector)
            weighted_sq_error = np.sum(np.nan_to_num(error_vector**2) * inv_cov_vector)
            exp_misfit = np.exp(
                np.clip(weighted_sq_error, -700, 700)
            )  # Clip to avoid overflow
            sum_exp_misfits_for_sample += p_i * exp_misfit

        # Calculate sample loss: -log(sum of weighted exponential misfits)
        if sum_exp_misfits_for_sample > 1e-300:  # Avoid log(0)
            sample_losses[i] = -np.log(sum_exp_misfits_for_sample)
        else:  # Assign large loss if sum is effectively zero
            sample_losses[i] = 700.0
    mean_nll = float(np.mean(sample_losses))
    return mean_nll


def calibrate_nll_variance_scale(
    y_val: np.ndarray, val_reals_orig: np.ndarray, app_config: AppConfig
) -> Tuple[float, float]:
    """Optimizes variance scale for heuristic realizations to minimize NLL.

    Args:
        y_val: Validation true values (n_samples, n_outputs).
        val_reals_orig: Original (unscaled by this function) heuristic
                        realizations from realization_generation module,
                        shape (n_samples, n_realizations, n_outputs).
                        Realization 0 is assumed to be the base prediction.
        app_config: Application configuration object.

    Returns:
        A tuple (best_scale, best_nll) for the heuristic realizations.
    """
    cfg_rc = app_config.realization_calib
    logger.info("Optimizing heuristic realization scale factor for NLL...")

    # Realization 0 is the base mean prediction; others are deviations from it.
    base_pred = val_reals_orig[:, 0, :].copy()
    best_scale = cfg_rc.nll_calibration_initial_scale

    # Calculate NLL with the initial scale factor
    current_reals_scaled = val_reals_orig.copy()
    for r_idx in range(1, current_reals_scaled.shape[1]):  # Scale rlz 1 to N-1
        deviation = val_reals_orig[:, r_idx, :] - base_pred
        current_reals_scaled[:, r_idx, :] = base_pred + (deviation * best_scale)
    best_nll = calculate_nll_loss(y_val, current_reals_scaled, app_config)
    logger.info(f"NLL with initial scale {best_scale:.3f}: {best_nll:.4f}")

    # Define search range for the scale factor
    s_min = max(
        0.01, best_scale - cfg_rc.nll_calibration_search_range_factor * abs(best_scale)
    )
    s_max = best_scale + cfg_rc.nll_calibration_search_range_factor * abs(best_scale)
    s_max = max(s_min + 0.01, s_max)  # Ensure s_max > s_min

    test_scales = np.unique(
        np.clip(
            np.linspace(s_min, s_max, cfg_rc.nll_calibration_search_steps),
            0.01,  # Min reasonable scale
            3.0,  # Max reasonable scale
        )
    )
    logger.info(
        f"Search scale range for NLL calibration: ({s_min:.3f} - {s_max:.3f}), "
        f"evaluating {len(test_scales)} steps."
    )

    for scale_candidate in tqdm(
        test_scales, desc="OptimizeHeuristicScale", disable=len(test_scales) < 5
    ):
        if np.isclose(scale_candidate, best_scale):
            continue

        temp_reals_scaled = val_reals_orig.copy()
        for r_idx in range(1, temp_reals_scaled.shape[1]):
            deviation = val_reals_orig[:, r_idx, :] - base_pred
            temp_reals_scaled[:, r_idx, :] = base_pred + (deviation * scale_candidate)

        current_nll = calculate_nll_loss(y_val, temp_reals_scaled, app_config)

        if current_nll < best_nll:
            best_nll = current_nll
            best_scale = scale_candidate
            logger.debug(
                f"New optimal scale: {best_scale:.3f} -> Min Comp. NLL: {best_nll:.4f}"
            )

    logger.info(
        f"Heuristic scale optimization complete. Optimal scale: {best_scale:.3f}, "
        f"Min Comp. NLL: {best_nll:.4f}"
    )
    return best_scale, best_nll


def analyze_variance_structure(
    X_train_fold_enhanced: np.ndarray,
    y_train_fold: np.ndarray,
    num_derived_features: int,
    app_config: AppConfig,
) -> Dict[str, Any]:
    """Analyzes variance structure of training data for a fold.

    Computes input/output variances per position, mean autocorrelation
    distance, and NLL-derived variance scale factors for realizations.

    Args:
        X_train_fold_enhanced: Training features (original + derived).
        y_train_fold: Training targets.
        num_derived_features: Number of derived features in X_train_fold_enhanced.
        app_config: Application configuration.

    Returns:
        A dictionary containing various variance and autocorrelation metrics.
    """
    logger.info("Analyzing variance structure of training data for the current fold...")
    col_cfg: ColumnConfig = app_config.columns
    realiz_calib_cfg: RealizationCalibrationConfig = app_config.realization_calib

    n_original_features = X_train_fold_enhanced.shape[1] - num_derived_features
    if n_original_features < 0:
        raise ValueError(
            "num_derived_features > total features in X_train_fold_enhanced."
        )

    X_original_part = X_train_fold_enhanced[:, :n_original_features]
    n_samples_in_fold = X_original_part.shape[0]

    input_variances = np.var(X_original_part, axis=0)
    output_variances = np.var(y_train_fold, axis=0)

    autocorrelation_distances: List[float] = []
    # Limit samples for ACF calculation for efficiency
    num_samples_for_autocorr = min(500, n_samples_in_fold)

    if num_samples_for_autocorr > 0:
        indices_for_autocorr = np.random.choice(
            n_samples_in_fold, num_samples_for_autocorr, replace=False
        )
        for i in indices_for_autocorr:
            series = X_original_part[i, :]
            # Use the corrected public function from feature_engineering
            valid_series_points, _ = get_valid_points_for_series(series)

            if len(valid_series_points) < 20:  # Need sufficient points for ACF
                continue
            try:
                centered_series = valid_series_points - np.mean(valid_series_points)
                autocorr_full = np.correlate(
                    centered_series, centered_series, mode="full"
                )

                acf_at_lag_zero = autocorr_full[len(centered_series) - 1]
                if abs(acf_at_lag_zero) < 1e-9:  # Avoid division by zero
                    continue

                normalized_autocorr = autocorr_full / acf_at_lag_zero
                # Positive lags only (middle to end of 'full' output)
                positive_lags_acf = normalized_autocorr[len(centered_series) - 1 :]

                # Find first lag where ACF drops below 0.5
                crossing_indices = np.where(positive_lags_acf < 0.5)[0]
                if len(crossing_indices) > 0:
                    # crossing_indices[0] is the first lag (0-indexed from start of positive_lags_acf)
                    autocorrelation_distances.append(float(crossing_indices[0]))
            except Exception as e_acf:
                logger.trace(
                    f"Autocorrelation calculation failed for sample {i}: {e_acf}"
                )

    mean_autocorr_dist = (
        np.mean(autocorrelation_distances)
        if autocorrelation_distances
        else 5.0  # Default
    )
    logger.info(
        f"Estimated mean autocorrelation distance (ACF < 0.5): "
        f"{mean_autocorr_dist:.2f} lags/positions."
    )

    # NLL-based variance scaling factors for realization generation
    inv_cov_vector = get_inverse_covariance_vector(
        col_cfg.n_outputs,
        realiz_calib_cfg.log_slopes,
        realiz_calib_cfg.log_offsets,
    )
    # Scale factor is 1/sqrt(inv_cov), then normalized by its mean
    variance_scales_for_rz = 1.0 / np.sqrt(inv_cov_vector + 1e-9)
    mean_rz_scale = np.mean(variance_scales_for_rz)
    if mean_rz_scale > 1e-9:
        variance_scales_for_rz /= mean_rz_scale
    else:  # Fallback if mean scale is near zero
        variance_scales_for_rz = np.ones_like(inv_cov_vector)

    return {
        "input_variance_per_pos": input_variances,
        "output_variance_per_pos": output_variances,
        "mean_autocorrelation_dist": mean_autocorr_dist,
        "variance_scale_factors_realization": variance_scales_for_rz,
    }
