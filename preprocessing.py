"""Handles data preprocessing including log transformation and feature engineering orchestration.

This module performs critical data preparation steps:
- Selection of input features (X_raw).
- Shifting of X_raw to ensure all values are positive for robust log transform.
- Log transformation of the shifted X_raw.
- Outlier capping on the log-transformed data.
- Imputation of missing values using KNNImputer.
- Orchestration of derived feature creation by calling the feature_engineering module.
- Selection of target variables (y_targets).
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd  # Required for DataFrame type hint and operations
from loguru import logger
from sklearn.impute import KNNImputer

from config import AppConfig  # For accessing various configuration sub-models
import feature_engineering  # To call the main feature creation function


def _cap_outliers(arr: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Performs column-wise percentile capping on a NumPy array.

    NaN values in the input array are ignored during percentile calculation
    and are preserved in the output. If a column is all NaNs, or if
    percentiles result in NaNs (e.g., empty column slice after NaNs),
    that column is returned unchanged.

    Args:
        arr: The 2D NumPy array to cap. Each column is treated independently.
        low_pct: The lower percentile (e.g., 1.0 for 1st percentile).
        high_pct: The upper percentile (e.g., 99.0 for 99th percentile).

    Returns:
        A new NumPy array with outliers capped. If input was 1D, output is 1D.
    """
    if arr.ndim == 1:  # Handle 1D array by temporarily making it 2D
        input_arr_2d = arr.reshape(-1, 1)
        was_1d_input = True
    else:
        input_arr_2d = arr
        was_1d_input = False

    capped_arr = input_arr_2d.copy()  # Work on a copy

    for col_idx in range(capped_arr.shape[1]):
        column_data_slice = capped_arr[:, col_idx]  # This is a view

        # Check if the column slice is entirely NaNs
        if np.all(np.isnan(column_data_slice)):
            # logger.trace(f"Column {col_idx} is all NaNs. Skipping capping.")
            continue  # Leave column as is

        # Calculate percentiles ignoring NaNs
        # np.nanpercentile handles arrays with NaNs gracefully.
        lower_bound = np.nanpercentile(column_data_slice, low_pct)
        upper_bound = np.nanpercentile(column_data_slice, high_pct)

        # If bounds are NaN (e.g., column was all NaNs after all, or empty slice)
        # This shouldn't happen if np.all(np.isnan) check passed, but defensive.
        if np.isnan(lower_bound) or np.isnan(upper_bound):
            logger.trace(
                f"Column {col_idx}: Could not determine capping bounds "
                f"(likely due to excessive NaNs). Skipping cap for this column."
            )
            continue

        # np.clip handles NaNs correctly by leaving them as NaNs
        capped_arr[:, col_idx] = np.clip(column_data_slice, lower_bound, upper_bound)

    return capped_arr.flatten() if was_1d_input else capped_arr


def prepare_features_and_target(
    df: pd.DataFrame,
    imputer: KNNImputer,  # Pass the imputer instance for test set consistency
    app_config: AppConfig,
    *,  # Force subsequent arguments to be keyword-only for clarity
    is_fitting_imputer: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Prepares features (X_enhanced) and target (y_targets) from a DataFrame.

    The primary steps involved are:
    1.  Selection of input features (X_raw) based on configuration.
    2.  Shifting X_raw values to ensure they are all positive before log transform.
    3.  Applying log transformation (np.log) to the shifted data.
    4.  Capping outliers on the log-transformed data using percentiles.
    5.  Imputing any remaining missing values using the provided KNNImputer.
    6.  Generating derived features using the `feature_engineering` module.
    7.  Selecting and returning target variables (if applicable, e.g., for training).

    Args:
        df: The input pandas DataFrame (either training or test data).
        imputer: A scikit-learn KNNImputer instance.
        app_config: The main application configuration object.
        is_fitting_imputer: Boolean flag indicating whether to fit the imputer
                            (True for training data) or only transform
                            (False for test/validation data).

    Returns:
        A tuple containing:
            - X_enhanced (np.ndarray): The final feature matrix, combining
              processed original features and derived features.
            - y_targets (Optional[np.ndarray]): The target variable matrix,
              or None if not applicable (e.g., for test data).
            - n_derived_actual (int): The actual number of derived features
              generated by the feature engineering step.

    Raises:
        ValueError: If required input or output columns are missing from the DataFrame
                    when they are expected (e.g., output columns during training).
    """
    cfg_cols = app_config.columns
    cfg_derived = app_config.derived_features

    action_description = (
        "Fitting imputer & preparing features"
        if is_fitting_imputer
        else "Preparing features (using existing imputer)"
    )
    logger.info(action_description)

    # --- 1. Select Input Features (X_raw) ---
    # Ensure all configured input columns are present in the DataFrame
    missing_input_cols = [col for col in cfg_cols.input_cols if col not in df.columns]
    if missing_input_cols:
        err_msg = f"Required input columns missing from DataFrame: {missing_input_cols}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    X_raw = df[cfg_cols.input_cols].to_numpy(dtype=float)
    logger.debug(
        f"Raw X data shape: {X_raw.shape}. "
        f"Min: {np.nanmin(X_raw):.2f}, Max: {np.nanmax(X_raw):.2f}"
    )

    # --- 2. Shift and Log Transform Input Features ---
    # Ensure all non-NaN values are positive before applying np.log
    X_for_log_transform = X_raw.copy()  # Work on a copy
    # Calculate minimum non-NaN value to determine necessary shift
    # Handle cases where X_for_log_transform might be all NaNs after column exclusion
    if X_for_log_transform.size == 0:  # If no input columns remained
        logger.warning(
            "X_raw is empty (no input columns). Proceeding with empty X_log_transformed."
        )
        X_log_transformed = np.empty((X_raw.shape[0], 0))  # Maintain sample dimension
    elif np.all(np.isnan(X_for_log_transform)):
        logger.warning("X_raw is all NaNs. Log transform will result in all NaNs.")
        X_log_transformed = X_for_log_transform  # Already all NaNs
    else:
        min_val_overall = np.nanmin(X_for_log_transform)
        target_min_after_shift = 1e-6  # Small positive value to avoid log(0)

        if not np.isnan(min_val_overall) and min_val_overall <= target_min_after_shift:
            # Shift needed to make all values > 0 (specifically > target_min_after_shift)
            shift_value = target_min_after_shift - min_val_overall
            logger.info(
                f"Shifting X_raw by {shift_value:.4f} (min_val: {min_val_overall:.2f}) "
                f"to ensure all values are > 0 before log."
            )
            # Apply shift only to non-NaN values; NaNs remain NaNs
            non_nan_mask = ~np.isnan(X_raw)
            X_for_log_transform[non_nan_mask] = X_raw[non_nan_mask] + shift_value
        # Else (min_val_overall is NaN or > target_min_after_shift), no shift needed

        logger.info(
            "Applying np.log transformation to (potentially shifted) input features."
        )
        with warnings.catch_warnings():  # Suppress RuntimeWarning for log of zero/negative
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X_log_transformed = np.log(X_for_log_transform)  # Original NaNs propagate

        # Handle any infinities created by log (e.g., log(0) if shift resulted in exact zero)
        if np.isinf(X_log_transformed).any():
            num_infs = np.isinf(X_log_transformed).sum()
            logger.warning(
                f"{num_infs} infinities found after log transform; converting to NaN."
            )
            X_log_transformed[np.isinf(X_log_transformed)] = np.nan

    logger.debug(
        f"X_log_transformed data shape: {X_log_transformed.shape}, "
        f"Total NaNs after log: {np.isnan(X_log_transformed).sum()}"
    )

    # --- 3. Outlier Capping (on log-transformed data) ---
    if X_log_transformed.shape[1] > 0:  # Only cap if there are columns
        logger.info(
            "Applying outlier capping to log-transformed features at "
            "[1.0th, 99.0th] percentiles (column-wise)."
        )
        X_capped = _cap_outliers(X_log_transformed, 1.0, 99.0)
    else:  # No columns to cap (e.g., if all input columns were excluded)
        X_capped = X_log_transformed  # Which is an empty array preserving num_samples
    logger.debug(f"X_capped data shape: {X_capped.shape}")

    # --- 4. Imputation ---
    if X_capped.shape[1] > 0:  # Only impute if there are columns
        logger.info(
            f"{'Fitting and transforming' if is_fitting_imputer else 'Transforming'} "
            "with KNNImputer."
        )
        if is_fitting_imputer:
            X_imputed = imputer.fit_transform(X_capped)
        else:
            X_imputed = imputer.transform(X_capped)
    else:  # No columns to impute
        X_imputed = X_capped  # Remains empty preserving num_samples
    logger.debug(f"X_imputed data shape: {X_imputed.shape}")

    if (
        np.isnan(X_imputed).any() or np.isinf(X_imputed).any()
    ):  # Should not happen after KNN
        logger.warning(
            "NaNs/Infs detected post-imputation. This is unexpected after KNN. "
            "Applying np.nan_to_num (filling with 0)."
        )
        X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 5. Feature Engineering ---
    logger.info(f"Creating {cfg_derived.total_n_derived_features} derived features...")
    # X_imputed now contains log-transformed, capped, and imputed original features
    X_derived = feature_engineering.create_all_derived_features(
        X_imputed=X_imputed,
        derived_feat_cfg=cfg_derived,
        col_cfg=cfg_cols,  # ColumnConfig still needed for some feature_engineering logic (e.g., x-value ranges)
    )
    n_derived_actual = X_derived.shape[1]
    logger.debug(f"X_derived data shape: {X_derived.shape}")

    if n_derived_actual != cfg_derived.total_n_derived_features:
        logger.critical(
            f"Derived feature count mismatch: Expected {cfg_derived.total_n_derived_features}, "
            f"Got {n_derived_actual}. Attempting to fix dimensions."
        )
        # Defensive padding/truncating
        if n_derived_actual > cfg_derived.total_n_derived_features:
            X_derived = X_derived[:, : cfg_derived.total_n_derived_features]
        else:
            padding_shape = (
                X_derived.shape[0],
                cfg_derived.total_n_derived_features - n_derived_actual,
            )
            padding = np.zeros(padding_shape)
            X_derived = np.hstack((X_derived, padding))
        n_derived_actual = X_derived.shape[1]  # Update actual count post-fix

    X_enhanced = np.hstack((X_imputed, X_derived))
    logger.info(
        f"Enhanced features created. Shape: {X_enhanced.shape} "
        f"(Input: {X_imputed.shape[1]}, Derived: {X_derived.shape[1]})"
    )

    # --- 6. Select Target Variable (y_targets) ---
    y_targets: Optional[np.ndarray] = None
    if cfg_cols.output_cols:
        missing_output_cols = [c for c in cfg_cols.output_cols if c not in df.columns]
        if is_fitting_imputer and missing_output_cols:
            err_msg = f"Missing output columns in training data: {missing_output_cols}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if not missing_output_cols:  # All output columns present
            y_targets = df[cfg_cols.output_cols].to_numpy(dtype=float)
            logger.info(f"Targets extracted. Shape: {y_targets.shape}")
        elif not is_fitting_imputer:  # Test data, no error if output cols missing
            logger.info(
                "Output columns not found or not expected (e.g., test data). y_targets is None."
            )
    else:  # No output_cols defined in config
        logger.info("No output columns defined in config. y_targets is None.")

    return X_enhanced, y_targets, n_derived_actual
