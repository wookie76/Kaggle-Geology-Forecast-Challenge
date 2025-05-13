"""Handles Stage 1 modeling using XGBoost and RandomForest ensembles.

This module includes functionalities for:
- Hyperparameter Optimization (HPO) using BayesSearchCV.
- Averaging HPO parameters across cross-validation folds.
- Training ensemble models (XGBoost and RandomForest) for specific output
  positions within each fold, parallelized using joblib.
- Predicting a base mean (R0) by ensembling predictions from these models
  and interpolating across output positions.
"""

import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed, dump, load
from loguru import logger
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

from config import AppConfig, GeneralConfig, ModelHyperparametersConfig
from constants import ModelType, OptimizedModelParams

# Type alias for arguments passed to the model training worker function.
TrainWorkerArgs = Tuple[
    int,  # target_pos_idx (0-based)
    str,  # X_mmap_path_str
    str,  # y_mmap_path_str
    Dict[str, Any],  # xgb_hpo_params_fold
    Dict[str, Any],  # rf_hpo_params_fold
    int,  # worker_base_seed
]


def optimize_hyperparameters(
    X_train_fold_processed: np.ndarray,
    y_target_for_pos: np.ndarray,
    model_type: ModelType,
    model_hparams_cfg: ModelHyperparametersConfig,
    general_cfg: GeneralConfig,
    base_hpo_seed: int,
) -> Dict[str, Any]:
    """Optimizes hyperparameters for a model using BayesSearchCV.

    Args:
        X_train_fold_processed: Preprocessed training features for the fold.
        y_target_for_pos: Target values for a specific output position.
        model_type: The type of model (XGBOOST or RANDOM_FOREST).
        model_hparams_cfg: Configuration for model hyperparameters and HPO.
        general_cfg: General pipeline configuration (e.g., num_cores).
        base_hpo_seed: Base seed for reproducibility of HPO.

    Returns:
        A dictionary containing the best hyperparameters found.
    """
    model_name_str = model_type.name.replace("_", " ").title()
    logger.info(
        f"Optimizing {model_name_str} with BayesSearchCV "
        f"(N_iter={model_hparams_cfg.bayes_search_n_iter}, "
        f"CV_folds={model_hparams_cfg.bayes_search_cv_folds})..."
    )

    param_space_to_use = (
        model_hparams_cfg.xgb_param_space
        if model_type == ModelType.XGBOOST
        else model_hparams_cfg.rf_param_space
    )

    base_estimator: Union[xgb.XGBRegressor, RandomForestRegressor]
    if model_type == ModelType.XGBOOST:
        # n_jobs=1 for XGBoost during HPO when BayesSearchCV itself is parallelized
        base_estimator = xgb.XGBRegressor(random_state=base_hpo_seed, n_jobs=1)
    elif model_type == ModelType.RANDOM_FOREST:
        # n_jobs=1 for RandomForest during HPO for similar reasons
        base_estimator = RandomForestRegressor(random_state=base_hpo_seed, n_jobs=1)
    else:
        raise ValueError(f"Unsupported model type for HPO: {model_type}")

    # BayesSearchCV will use general_cfg.num_cores for its internal CV parallelism
    n_jobs_bayes = general_cfg.num_cores

    optimizer = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=param_space_to_use,
        n_iter=model_hparams_cfg.bayes_search_n_iter,
        cv=KFold(
            n_splits=model_hparams_cfg.bayes_search_cv_folds,
            shuffle=True,
            random_state=base_hpo_seed + 1,  # Seed for CV splits
        ),
        scoring="neg_mean_squared_error",
        random_state=base_hpo_seed + 2,  # Seed for BayesSearch stochasticity
        n_jobs=n_jobs_bayes,
        verbose=0,  # Can be increased for more HPO verbosity
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        optimizer.fit(X_train_fold_processed, y_target_for_pos)

    best_params = dict(optimizer.best_params_)
    best_params["random_state"] = base_hpo_seed  # Use HPO seed for final model

    # Set n_jobs for the final model instantiation based on convention
    if model_type == ModelType.XGBOOST:
        best_params["n_jobs"] = 1  # XGBoost typically managed by outer parallelism
    elif model_type == ModelType.RANDOM_FOREST:
        best_params["n_jobs"] = -1  # RandomForest can parallelize internally

    logger.info(
        f"Best {model_name_str} params: {best_params} "
        f"(CV Score: {optimizer.best_score_:.4f})"
    )
    return best_params


def average_hpo_params(
    fold_hpo_results: List[Dict[str, Any]],
    default_params: Dict[str, Any],
    *,
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    """Averages numeric HPO parameters across folds.

    For non-numeric parameters, the value from the first fold's results
    is used. If a parameter is not found in HPO results, its value from
    `default_params` is retained.

    Args:
        fold_hpo_results: A list of HPO result dictionaries, one for each fold.
        default_params: Default parameters for the model.
        model_name: Name of the model (e.g., "XGBoost") for logging.
        seed: Random seed to set in the final averaged parameters.

    Returns:
        A dictionary of final averaged (or selected) hyperparameters.
    """
    if not fold_hpo_results or not fold_hpo_results[0]:
        logger.warning(
            f"No HPO results provided for {model_name}; using default parameters."
        )
        final_params = default_params.copy()
        final_params["random_state"] = seed
        return final_params

    final_params = default_params.copy()
    all_param_keys_from_hpo = set()
    for res in fold_hpo_results:
        all_param_keys_from_hpo.update(res.keys())

    for key in all_param_keys_from_hpo:
        if key in ("random_state", "n_jobs"):  # Handled explicitly later
            continue

        # Attempt to average numeric parameters
        numeric_values_for_key = [
            res[key]
            for res in fold_hpo_results
            if key in res and isinstance(res[key], (int, float))
        ]

        if numeric_values_for_key:
            if isinstance(numeric_values_for_key[0], int):
                final_params[key] = int(round(np.mean(numeric_values_for_key)))
            else:  # float
                final_params[key] = float(np.mean(numeric_values_for_key))
        elif key in fold_hpo_results[0]:  # Non-numeric, take from first fold
            final_params[key] = fold_hpo_results[0][key]
        # If key is missing from first fold but present in others (and non-numeric),
        # it would not be picked up here. Relies on default_params for coverage.

    # Ensure essential keys are set
    final_params["random_state"] = seed
    if model_name.lower() == "xgboost":
        final_params["n_jobs"] = default_params.get("n_jobs", 1)
    elif model_name.lower() == "randomforest":
        final_params["n_jobs"] = default_params.get("n_jobs", -1)

    # Ensure all keys from default_params are in final_params
    for key_default, val_default in default_params.items():
        final_params.setdefault(key_default, val_default)

    return final_params


def _train_single_model_worker(
    args: TrainWorkerArgs,
) -> Tuple[int, Dict[str, Any]]:
    """Trains XGBoost and RandomForest for a single output position.

    This worker function is designed to be run in parallel by joblib.
    Early stopping for XGBoost is omitted in this version to avoid
    `TypeError` previously encountered with `joblib` and XGBoost's
    `fit` parameters. The number of estimators is taken from HPO.

    Args:
        args: A tuple containing all necessary arguments for the worker.
            (target_pos_idx, X_mmap_path_str, y_mmap_path_str,
             xgb_hpo_params_fold, rf_hpo_params_fold, worker_base_seed)

    Returns:
        A tuple containing the 1-based target position label and a
        dictionary of trained models and their metadata.
    """
    (
        target_pos_idx,
        X_mmap_path_str,
        y_mmap_path_str,
        xgb_hpo_params_fold,
        rf_hpo_params_fold,
        worker_base_seed,
    ) = args

    # Load data from memory-mapped files
    X_fold_data: np.ndarray = load(X_mmap_path_str, mmap_mode="r")
    y_fold_data: np.ndarray = load(y_mmap_path_str, mmap_mode="r")

    current_target_series = y_fold_data[:, target_pos_idx]

    # XGBoost Model Training
    # Use n_estimators from HPO directly; no early stopping in this worker.
    xgb_n_estimators = xgb_hpo_params_fold.get(
        "n_estimators", 150
    )  # Default if not in HPO
    xgb_params_for_fit = {
        **xgb_hpo_params_fold,
        "n_estimators": xgb_n_estimators,
        "random_state": worker_base_seed + 10 + target_pos_idx,
    }
    xgb_model = xgb.XGBRegressor(**xgb_params_for_fit)
    with warnings.catch_warnings():  # Suppress fit-related warnings from XGBoost
        warnings.simplefilter("ignore")
        xgb_model.fit(X_fold_data, current_target_series, verbose=False)

    # Since no early stopping, best_iteration is n_estimators
    xgb_best_iteration: Optional[int] = xgb_n_estimators
    xgb_preds_train = xgb_model.predict(X_fold_data)
    mse_xgb_train = mean_squared_error(current_target_series, xgb_preds_train)

    # RandomForest Model Training
    rf_params_for_fit = {
        **rf_hpo_params_fold,
        "random_state": worker_base_seed + 20 + target_pos_idx,
    }
    rf_model = RandomForestRegressor(**rf_params_for_fit)
    rf_model.fit(X_fold_data, current_target_series)
    rf_preds_train = rf_model.predict(X_fold_data)
    mse_rf_train = mean_squared_error(current_target_series, rf_preds_train)

    # Calculate ensemble weights based on inverse MSE
    w_xgb_raw = 1.0 / (mse_xgb_train + 1e-9)  # Epsilon for numerical stability
    w_rf_raw = 1.0 / (mse_rf_train + 1e-9)
    total_weight_raw = w_xgb_raw + w_rf_raw
    weight_xgb = (w_xgb_raw / total_weight_raw) if total_weight_raw > 1e-9 else 0.5

    # Return 1-based label for dictionary key consistency
    model_label = target_pos_idx + 1
    return model_label, {
        "xgb_model": xgb_model,
        "rf_model": rf_model,
        "weight_xgb": weight_xgb,
        "weight_rf": 1.0 - weight_xgb,
        "mse_xgb_train": mse_xgb_train,
        "mse_rf_train": mse_rf_train,
        "xgb_best_iteration": xgb_best_iteration,
    }


def train_ensemble_models_for_fold(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    num_derived_features: int,
    opt_params: OptimizedModelParams,
    app_config: AppConfig,
    fold_idx: int,
) -> Tuple[Dict[int, Any], StandardScaler, List[int]]:
    """Trains Stage 1 ensemble models for a given cross-validation fold.

    Scales original features, trains models for key output positions in parallel,
    and returns the trained models, scaler, and key output indices.

    Args:
        X_train_fold: Training features for the fold (original + derived).
        y_train_fold: Training targets for the fold.
        num_derived_features: Number of derived features in X_train_fold.
        opt_params: Optimized hyperparameters for XGBoost and RandomForest.
        app_config: Application configuration.
        fold_idx: Index of the current fold (0-based).

    Returns:
        A tuple:
            - Dictionary of trained models (key: 1-based output position).
            - Fitted StandardScaler instance for original features.
            - List of 0-based key output indices for which models were trained.
    """
    fold_display_num = fold_idx + 1
    logger.info(
        f"[F{fold_display_num}] Training Stage 1 models. "
        f"X_shape: {X_train_fold.shape}, y_shape: {y_train_fold.shape}"
    )

    n_original_features = X_train_fold.shape[1] - num_derived_features
    if n_original_features < 0:
        raise ValueError(
            "num_derived_features is greater than total features in X_train_fold."
        )

    # Scale only the original features part of X_train_fold
    scaler = StandardScaler().fit(X_train_fold[:, :n_original_features])
    X_train_fold_processed = np.hstack(
        (
            scaler.transform(X_train_fold[:, :n_original_features]),
            X_train_fold[:, n_original_features:],  # Concatenate derived features
        )
    )

    # Determine key output positions for which to train sparse models
    num_total_output_positions = y_train_fold.shape[1]
    key_output_indices_set = set()
    # Logic for selecting sparse key output indices based on original script
    key_output_indices_set.update(range(0, min(60, num_total_output_positions), 10))
    key_output_indices_set.update(range(60, min(240, num_total_output_positions), 30))
    key_output_indices_set.update(range(240, num_total_output_positions, 20))
    if num_total_output_positions > 0:
        key_output_indices_set.add(num_total_output_positions - 1)  # Ensure last point

    key_out_indices_0based = sorted(
        [
            idx
            for idx in list(key_output_indices_set)
            if 0 <= idx < num_total_output_positions
        ]
    )

    if not key_out_indices_0based and num_total_output_positions > 0:
        logger.warning(
            f"[F{fold_display_num}] No valid key output indices derived for S1 "
            f"training, defaulting to [0]."
        )
        key_out_indices_0based = [0]
    elif not key_out_indices_0based:  # num_total_output_positions must be 0
        logger.warning(
            f"[F{fold_display_num}] No output positions available "
            f"(n_outputs=0). Skipping Stage 1 model training."
        )
        return {}, scaler, []

    trained_models: Dict[int, Any] = {}
    temp_dir_prefix = f"s1_fold{fold_display_num}_worker_data_"

    with tempfile.TemporaryDirectory(prefix=temp_dir_prefix) as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        X_mmap_path = temp_dir_path / "X_s1_processed_fold_data.mmap"
        y_mmap_path = temp_dir_path / "y_s1_targets_fold_data.mmap"

        dump(X_train_fold_processed, X_mmap_path)
        dump(y_train_fold, y_mmap_path)

        worker_base_seed = app_config.general.seed + fold_idx * 1000  # Seed per fold

        args_list_for_workers: List[TrainWorkerArgs] = [
            (
                target_pos_0based_idx,
                str(X_mmap_path),
                str(y_mmap_path),
                opt_params["xgb_params"],
                opt_params["rf_params"],
                worker_base_seed,
            )
            for target_pos_0based_idx in key_out_indices_0based
        ]

        logger.info(
            f"[F{fold_display_num}] Dispatching {len(args_list_for_workers)} "
            f"Stage 1 model training tasks to {app_config.general.num_cores} cores."
        )

        # Parallel execution of worker function
        # `backend="loky"` is robust for complex objects.
        results = Parallel(n_jobs=app_config.general.num_cores, backend="loky")(
            delayed(_train_single_model_worker)(worker_args)
            for worker_args in args_list_for_workers
        )

        for model_label, model_data in results:  # model_label is 1-based
            trained_models[model_label] = model_data
            logger.debug(
                f"[F{fold_display_num}] S1 Model {model_label}: "
                f"XGB_MSE={model_data['mse_xgb_train']:.3f} "
                f"(iters:{model_data.get('xgb_best_iteration', 'N/A')}), "
                f"RF_MSE={model_data['mse_rf_train']:.3f}, "
                f"XGB_Weight={model_data['weight_xgb']:.2f}"
            )

    logger.info(
        f"[F{fold_display_num}] Trained {len(trained_models)} Stage 1 "
        f"model pairs for key output positions."
    )
    return trained_models, scaler, key_out_indices_0based


def predict_with_ensemble(
    X_input: np.ndarray,
    models_dict: Dict[int, Any],  # Keys are 1-based output position labels
    scaler: StandardScaler,
    key_out_indices_0based: List[int],  # 0-based indices models were trained for
    num_derived_features: int,
    total_num_outputs: int,  # Desired length of the full R0 prediction vector
) -> np.ndarray:
    """Generates R0 (base mean prediction) using the Stage 1 ensemble.

    Predicts sparsely at `key_out_indices_0based` and interpolates these
    predictions to cover the `total_num_outputs` horizon.

    Args:
        X_input: Input features (original + derived).
        models_dict: Trained models from `train_ensemble_models_for_fold`.
        scaler: Fitted StandardScaler for original features.
        key_out_indices_0based: 0-based indices for which models are available.
        num_derived_features: Number of derived features in X_input.
        total_num_outputs: The total number of output steps to predict.

    Returns:
        A NumPy array of shape (n_samples, total_num_outputs) representing
        the interpolated R0 predictions.
    """
    num_samples = X_input.shape[0]
    num_original_features = X_input.shape[1] - num_derived_features

    if num_original_features < 0:
        raise ValueError(
            "num_derived_features is greater than total features in X_input."
        )

    if not models_dict or not key_out_indices_0based:
        logger.warning(
            "No models or key_out_indices provided for Stage 1 prediction. "
            "Returning array of zeros."
        )
        return np.zeros((num_samples, total_num_outputs))

    # Preprocess input: scale original features, concatenate derived
    X_input_processed = np.hstack(
        (
            scaler.transform(X_input[:, :num_original_features]),
            X_input[:, num_original_features:],
        )
    )

    # Store sparse predictions at key output positions
    key_predictions_sparse = np.full((num_samples, len(key_out_indices_0based)), np.nan)

    for i, k_idx_0based in enumerate(key_out_indices_0based):
        model_label = k_idx_0based + 1  # models_dict uses 1-based keys
        if model_label in models_dict:
            model_data = models_dict[model_label]
            xgb_preds = model_data["xgb_model"].predict(X_input_processed)
            rf_preds = model_data["rf_model"].predict(X_input_processed)
            key_predictions_sparse[:, i] = (model_data["weight_xgb"] * xgb_preds) + (
                model_data["weight_rf"] * rf_preds
            )
        # Else, slot remains NaN, handled during interpolation checks

    # Interpolate sparse predictions to the full output horizon
    all_predictions_interpolated = np.full((num_samples, total_num_outputs), np.nan)
    output_indices_for_interpolation = np.arange(total_num_outputs)

    # Use only valid (non-NaN) 0-based key indices for fitting spline
    # This should generally be all of them if models trained correctly for all keys
    x_known_coordinates = np.array(key_out_indices_0based)

    for i in range(num_samples):
        y_known_values_sample = key_predictions_sparse[i, :]

        # Filter out NaNs from y_known_values (e.g., if a model was missing)
        # and use corresponding x_known_coordinates
        valid_y_points_mask = ~np.isnan(y_known_values_sample)
        x_coords_for_fit = x_known_coordinates[valid_y_points_mask]
        y_values_for_fit = y_known_values_sample[valid_y_points_mask]

        if len(x_coords_for_fit) <= 1:  # Not enough points for robust interpolation
            fill_value = y_values_for_fit[0] if len(x_coords_for_fit) == 1 else 0.0
            all_predictions_interpolated[i, :] = fill_value
            continue

        # Determine spline degree (k)
        spline_k = min(3, len(x_coords_for_fit) - 1)  # k <= N-1 for N data points
        spline_k = max(1, spline_k)  # k must be at least 1

        # Adaptive smoothing factor for UnivariateSpline
        smoothing_s = 0.1  # Default
        if len(y_values_for_fit) > 2:
            differences = np.abs(np.diff(y_values_for_fit))
            mean_diff, max_diff = np.mean(differences), np.max(differences)
            if max_diff > 5 * mean_diff + 1e-6:
                smoothing_s = 0.01
            elif max_diff > 2 * mean_diff + 1e-6:
                smoothing_s = 0.05
            else:
                smoothing_s = 0.2
        else:  # Few points, less aggressive smoothing
            smoothing_s = 0.05

        try:
            interpolation_function = UnivariateSpline(
                x_coords_for_fit,
                y_values_for_fit,
                k=spline_k,
                s=smoothing_s,
                ext="const",  # Extrapolate with boundary values
            )
            all_predictions_interpolated[i, :] = interpolation_function(
                output_indices_for_interpolation
            )
        except Exception as e_spline:  # Broad exception for any spline error
            logger.warning(
                f"Spline interpolation failed for sample {i} "
                f"(N_points={len(x_coords_for_fit)}, k={spline_k}): {e_spline}. "
                "Falling back to linear interpolation (np.interp)."
            )
            # Fallback to simpler linear interpolation
            all_predictions_interpolated[i, :] = np.interp(
                output_indices_for_interpolation,
                x_coords_for_fit,
                y_values_for_fit,
            )

    # Fill any remaining NaNs (e.g., if all predictions for a sample were NaN initially)
    return np.nan_to_num(all_predictions_interpolated, nan=0.0)
