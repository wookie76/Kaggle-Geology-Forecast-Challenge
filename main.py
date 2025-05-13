"""Main pipeline orchestrator for the Geology Forecast Challenge.

Refactored (v2_pywan_refactor) for enhanced component separation,
improved performance through vectorized feature engineering (implemented
in feature_engineering.py), and robust handling of missing data in input
columns.
"""

import json
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

# --- SET MATPLOTLIB BACKEND ---
# This MUST be done BEFORE importing matplotlib.pyplot or any part of
# matplotlib that might implicitly initialize a backend.
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for saving plots to files
# --- END OF MATPLOTLIB BACKEND SETTING ---


from pathlib import Path

# --- Core Libraries ---
import numpy as np
import pandas as pd  # Needed for DataFrame operations after loading
from loguru import logger

# --- Scikit-learn ---
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
from config import AppConfig
import constants

# --- Project Modules (Refactored Structure) ---
import data_io
import preprocessing
from models import gru_refiner, stage1_ensemble
import realization_generation
import evaluation
import visualization


# --- Global Setup Functions ---


def setup_logging(log_file_pattern_template: str, script_version_tag: str) -> None:
    """Configures Loguru logger.

    Sets up logging to stderr and a file sink with rotation and retention.

    Args:
        log_file_pattern_template: String pattern for log file path, may contain {time}.
        script_version_tag: Tag to include in the log file name for identification.
    """
    logger.remove()  # Remove default handlers

    # Add handler for stderr
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <lvl>{level:<8}</lvl> | "
            "<cyan>{name}:{function}:{line}</cyan> - <lvl>{message}</lvl>"
        ),
    )

    # Add handler for log file
    # Loguru automatically replaces '{time}' in the sink path
    logger.add(
        log_file_pattern_template,
        level="DEBUG",  # Log DEBUG and above to file
        rotation="10 MB",  # New file every 10 MB
        retention="3 days",  # Keep logs for 3 days
        enqueue=True,  # Use a queue for safe multiprocessing logging
        format="{time} | {level:<8} | {module}:{function}:{line} - {message}",
    )
    logger.info(
        f"Logging initialized. Log file pattern (template): "
        f"{log_file_pattern_template}"
    )


def seed_everything(seed: int) -> None:
    """Sets random seeds for reproducibility across relevant libraries.

    Args:
        seed: The integer seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        # TensorFlow seed setting should happen after TensorFlow is imported
        # If TF is imported in models.gru_refiner, this block is fine here
        # as long as seed_everything is called after the first TF import in main_pipeline
        # or just before using seeded TF functions. Putting it here is safer.
        tf.random.set_seed(seed)
        logger.info(f"TensorFlow random seed set to: {seed}")
    except ImportError:
        logger.debug("TensorFlow not found, skipping TensorFlow seed setting.")
    logger.info(f"Global NumPy/Random/Hash seeds set to: {seed}")


# --- Main Pipeline Function ---
def main_pipeline(app_cfg: AppConfig) -> None:
    """Executes the full geology forecasting pipeline.

    This is the main orchestration function that calls different modules
    to perform data loading, preprocessing, modeling, evaluation, and submission.

    Args:
        app_cfg: The main AppConfig object containing all pipeline parameters.

    Raises:
        SystemExit: If critical data loading or processing steps fail.
        ValueError: If configuration validation fails unexpectedly during runtime.
    """
    logger.info(f"Starting Pipeline: {app_cfg.paths.script_version_tag}")
    logger.info(
        f"General Config: Seed={app_cfg.general.seed}, "
        f"Folds={app_cfg.general.n_folds}, "
        f"Cores={app_cfg.general.num_cores}"
    )

    # Log feature and column configurations
    if app_cfg.derived_features.colab:
        colab_n_feats_str = str(app_cfg.derived_features.colab.n_features)
    else:
        colab_n_feats_str = "N/A (Colab config not initialized)"

    logger.info(
        f"Feature Config: Original={app_cfg.derived_features.original.n_features}, "
        f"Wavelet={app_cfg.derived_features.wavelet.n_derived_features}, "
        f"Colab={colab_n_feats_str}, "
        f"Total Derived={app_cfg.derived_features.total_n_derived_features}"
    )
    # Note: Input Length and Num Outputs will be logged again after potential exclusion
    logger.info(
        f"Column Config (Initial): Input Len={app_cfg.columns.input_length}, "
        f"Num Outputs={app_cfg.columns.n_outputs}"
    )

    # Suppress specific warnings to keep logs cleaner during known operations
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.interpolate")
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning
    )  # e.g., np.nanmean of empty slice
    warnings.filterwarnings("ignore", category=FutureWarning)  # often from libraries

    # --- Step 1: Data Loading ---
    logger.info("--- Step 1: Data Loading ---")

    # Load all relevant numeric columns including potential inputs and outputs
    # for training data to analyze missingness.
    # Load only existing input columns for test data.
    train_df, train_ids = data_io.load_single_csv(
        file_path=app_cfg.paths.train_file,
        id_col_name=app_cfg.columns.geology_id_col,
        numeric_cols=app_cfg.columns.all_numeric_feature_cols,
        dataset_description="Training",
    )
    test_df, test_ids = data_io.load_single_csv(
        file_path=app_cfg.paths.test_file,
        id_col_name=app_cfg.columns.geology_id_col,
        numeric_cols=app_cfg.columns.input_cols,  # Load initial input cols for test
        dataset_description="Test",
    )

    if train_df is None or train_ids is None:
        logger.critical("Training data load failed. Exiting.")
        sys.exit(1)

    # --- Step 1b: Analyze and Exclude Input Columns based on Missingness ---
    logger.info(
        "--- Step 1b: Analyze and Exclude Input Columns based on Missingness ---"
    )

    # Identify input columns older than -49 (based on the range in config)
    input_cols_current = list(app_cfg.columns.input_cols)  # Work on a copy initially

    # Find columns in the current input list that are numerically less than -49
    cols_to_potentially_exclude = [
        col
        for col in input_cols_current
        if col.isdigit()
        or (
            col.startswith("-") and col[1:].isdigit()
        )  # Ensure it's a number-like string
        if int(col)
        < -49  # Exclude columns with names representing positions older than -49
    ]

    if not cols_to_potentially_exclude:
        logger.info("No input columns older than -49 found based on naming convention.")
        # Proceed without exclusion
    else:
        # Calculate missingness for these candidates in the training data
        # Ensure columns exist in the loaded train_df before accessing
        cols_in_train_df = [
            col for col in cols_to_potentially_exclude if col in train_df.columns
        ]

        if cols_in_train_df:
            missing_info_candidates = train_df[cols_in_train_df].isnull().sum()
            total_train_rows = len(train_df)

            # Get exclusion threshold from config or use a default
            # Add this threshold to AppConfig.general or ColumnConfig if needed
            missing_pct_threshold = 57.5  # Example threshold (make configurable)
            logger.info(
                f"Excluding input columns older than -49 with > {missing_pct_threshold}% missing data."
            )

            cols_to_exclude_final = [
                col
                for col in cols_in_train_df
                if (missing_info_candidates[col] / total_train_rows) * 100.0
                > missing_pct_threshold
            ]

            if cols_to_exclude_final:
                logger.info(
                    f"Excluding {len(cols_to_exclude_final)} input columns: {cols_to_exclude_final}"
                )

                # Update app_config.columns.input_cols and input_length
                app_cfg.columns.input_cols = [
                    col
                    for col in input_cols_current
                    if col not in cols_to_exclude_final
                ]
                app_cfg.columns.input_length = len(
                    app_cfg.columns.input_cols
                )  # Update derived length

                # Filter the DataFrames based on the new input_cols list
                # Keep ID + new input_cols + output_cols (for train)
                kept_cols_train = (
                    [app_cfg.columns.geology_id_col]
                    + app_cfg.columns.input_cols
                    + app_cfg.columns.output_cols
                )
                # Keep ID + new input_cols (for test)
                kept_cols_test = [
                    app_cfg.columns.geology_id_col
                ] + app_cfg.columns.input_cols

                train_df = train_df[kept_cols_train].copy()
                if test_df is not None:
                    test_df = test_df[kept_cols_test].copy()

                logger.info(f"Updated train_df shape after exclusion: {train_df.shape}")
                if test_df is not None:
                    logger.info(
                        f"Updated test_df shape after exclusion: {test_df.shape}"
                    )

            else:
                logger.info(
                    "No input columns older than -49 met the exclusion threshold."
                )
        else:
            logger.warning(
                "Potential input columns older than -49 not found in training data."
            )

    logger.info(
        f"Column Config (After Exclusion): Input Len={app_cfg.columns.input_length}, "
        f"Num Outputs={app_cfg.columns.n_outputs}"
    )

    # --- 2. Data Preprocessing & Feature Engineering ---
    logger.info("--- Step 2: Data Preprocessing & Feature Engineering ---")
    knn_imputer_main = KNNImputer(
        n_neighbors=app_cfg.model_hparams.knn_n_neighbors, weights="distance"
    )

    # prepare_features_and_target now operates on the potentially reduced DataFrames
    # and uses the updated app_cfg.columns.input_cols
    X_full_train_enhanced, y_full_train, num_derived_actual = (
        preprocessing.prepare_features_and_target(
            df=train_df,
            imputer=knn_imputer_main,
            app_config=app_cfg,  # Pass the full app_cfg for access to columns, derived_features
            is_fitting_imputer=True,
        )
    )

    if y_full_train is None:
        logger.critical(
            "Target variable (y_full_train) is None after preprocessing. Aborting."
        )
        sys.exit(1)
    if num_derived_actual != app_cfg.derived_features.total_n_derived_features:
        # This check relies on feature_engineering correctly implementing the expected count
        logger.critical(
            f"Derived feature count mismatch. Expected {app_cfg.derived_features.total_n_derived_features}, "
            f"Got {num_derived_actual}. Aborting."
        )
        sys.exit(1)
    logger.info(
        f"Full training data prepared: X_enhanced={X_full_train_enhanced.shape}, "
        f"y={y_full_train.shape}, Num_Derived_Features={num_derived_actual}"
    )

    # --- 3. Cross-Validation and Stage 1 Model Training (XGB/RF) ---
    logger.info("--- Step 3: Cross-Validation & Stage 1 Model Training ---")
    kfold = KFold(
        n_splits=app_cfg.general.n_folds,
        shuffle=True,
        random_state=app_cfg.general.seed,
    )

    fold_nll_scores: List[float] = []
    fold_opt_scales: List[float] = []
    fold_s1_xgb_hpo_params: List[Dict[str, Any]] = []
    fold_s1_rf_hpo_params: List[Dict[str, Any]] = []
    fold_gru_val_loss: List[float] = []

    X_val_last: Optional[np.ndarray] = None
    y_val_last: Optional[np.ndarray] = None

    for fold_idx, (train_indices, val_indices) in enumerate(
        kfold.split(X_full_train_enhanced, y_full_train)
    ):
        fold_display_num = fold_idx + 1
        logger.info(f"--- CV Fold {fold_display_num}/{app_cfg.general.n_folds} ---")

        X_tr, y_tr = X_full_train_enhanced[train_indices], y_full_train[train_indices]
        X_v, y_v = X_full_train_enhanced[val_indices], y_full_train[val_indices]

        if fold_idx == app_cfg.general.n_folds - 1:
            X_val_last, y_val_last = X_v.copy(), y_v.copy()

        # --- 3a. Analyze Variance Structure (per fold on training part) ---
        var_analysis_params_fold = evaluation.analyze_variance_structure(
            X_train_fold_enhanced=X_tr,
            y_train_fold=y_tr,
            num_derived_features=num_derived_actual,
            app_config=app_cfg,  # Needs column info, NLL calib params
        )

        # --- 3b. Stage 1 HPO and Model Training (XGB/RF Ensemble) ---
        # The scaler for HPO should use only the *original* features of the fold
        num_original_features_fold = X_tr.shape[1] - num_derived_actual
        scaler_hpo = StandardScaler().fit(X_tr[:, :num_original_features_fold])
        X_tr_processed_hpo = np.hstack(
            (
                scaler_hpo.transform(X_tr[:, :num_original_features_fold]),
                X_tr[
                    :, num_original_features_fold:
                ],  # Derived features bypass this scaler
            )
        )

        hpo_target_column_idx = app_cfg.columns.n_outputs // 2
        hpo_seed_fold_offset = fold_idx * 100

        xgb_hpo_p = stage1_ensemble.optimize_hyperparameters(
            X_train_fold_processed=X_tr_processed_hpo,
            y_target_for_pos=y_tr[:, hpo_target_column_idx],
            model_type=constants.ModelType.XGBOOST,
            model_hparams_cfg=app_cfg.model_hparams,
            general_cfg=app_cfg.general,
            base_hpo_seed=app_cfg.general.seed + hpo_seed_fold_offset + 300,
        )
        fold_s1_xgb_hpo_params.append(xgb_hpo_p)

        rf_hpo_p = stage1_ensemble.optimize_hyperparameters(
            X_train_fold_processed=X_tr_processed_hpo,
            y_target_for_pos=y_tr[:, hpo_target_column_idx],
            model_type=constants.ModelType.RANDOM_FOREST,
            model_hparams_cfg=app_cfg.model_hparams,
            general_cfg=app_cfg.general,
            base_hpo_seed=app_cfg.general.seed + hpo_seed_fold_offset + 400,
        )
        fold_s1_rf_hpo_params.append(rf_hpo_p)

        s1_cv_optimized_params_fold: constants.OptimizedModelParams = {  # type: ignore
            "xgb_params": xgb_hpo_p,
            "rf_params": rf_hpo_p,
        }  # type: ignore

        s1_models_fold, s1_scaler_fold, s1_key_out_indices_fold = (
            stage1_ensemble.train_ensemble_models_for_fold(
                X_train_fold=X_tr,  # Pass original X_tr
                y_train_fold=y_tr,
                num_derived_features=num_derived_actual,
                opt_params=s1_cv_optimized_params_fold,
                app_config=app_cfg,  # Needed for num_cores, seed
                fold_idx=fold_idx,
            )
        )

        # --- 3c. Stage 1 Prediction (R0 - Base Mean Prediction) ---
        R0_tr_s1 = stage1_ensemble.predict_with_ensemble(
            X_input=X_tr,
            models_dict=s1_models_fold,
            scaler=s1_scaler_fold,
            key_out_indices_0based=s1_key_out_indices_fold,
            num_derived_features=num_derived_actual,
            total_num_outputs=app_cfg.columns.n_outputs,
        )
        R0_v_s1 = stage1_ensemble.predict_with_ensemble(
            X_input=X_v,
            models_dict=s1_models_fold,
            scaler=s1_scaler_fold,
            key_out_indices_0based=s1_key_out_indices_fold,
            num_derived_features=num_derived_actual,
            total_num_outputs=app_cfg.columns.n_outputs,
        )

        # --- 3d. Stage 2 GRU Refiner Training & Prediction (per fold) ---
        R0_tr_gru_in = np.expand_dims(R0_tr_s1, axis=-1)
        R0_v_gru_in = np.expand_dims(R0_v_s1, axis=-1)

        gru_model_fold = gru_refiner.build_deterministic_gru_refiner(
            sequence_length=app_cfg.columns.n_outputs,
            num_features_in=1,
            num_features_out=1,
            model_hparams=app_cfg.model_hparams,
        )
        gru_history = gru_refiner.train_deterministic_gru_refiner(
            model=gru_model_fold,
            X_train_r0=R0_tr_gru_in,
            y_train_true=y_tr,
            X_val_r0=R0_v_gru_in,
            y_val_true=y_v,
            model_hparams=app_cfg.model_hparams,
            # epochs override from AppConfig.model_hparams.gru_epochs
        )
        if gru_history.history and "val_loss" in gru_history.history:
            fold_gru_val_loss.append(min(gru_history.history["val_loss"]))
        elif "loss" in gru_history.history:  # Fallback if no val data/loss
            fold_gru_val_loss.append(gru_history.history["loss"][-1])

        val_r0_refined_squeezed = gru_refiner.predict_with_deterministic_gru_refiner(
            model=gru_model_fold,
            X_r0_sequences=R0_v_gru_in,
            batch_size=app_cfg.model_hparams.gru_batch_size,
        )

        # --- 3e. Generate Heuristic Realizations & Calibrate NLL Scale (per fold) ---
        # generate_calibrated_realizations returns UNCALIBRATED heuristic realizations
        # The calibration happens *in evaluation.calibrate_nll_variance_scale*
        # based on these generated realizations and the true y_val.
        val_reals_heuristic_unscaled = realization_generation.generate_calibrated_realizations(
            X_full_features=X_v,  # Full features of validation set (original+derived)
            base_preds=val_r0_refined_squeezed,  # GRU-refined base prediction for validation
            var_analysis_params=var_analysis_params_fold,  # Variance params from *this fold's* training data
            num_derived_features=num_derived_actual,
            app_config=app_cfg,  # Needed for config access in worker (wavelet, calib, general)
            master_seed_offset=fold_idx,  # Different seed offset per fold's generation
        )
        # Calibrate NLL variance scale factor using the generated realizations
        opt_scale_fold, nll_comp_fold = evaluation.calibrate_nll_variance_scale(
            y_val=y_v, val_reals_orig=val_reals_heuristic_unscaled, app_config=app_cfg
        )
        fold_opt_scales.append(opt_scale_fold)
        fold_nll_scores.append(nll_comp_fold)

        logger.info(
            f"--- Fold {fold_display_num} Done: Scale={opt_scale_fold:.3f}, CompNLL={nll_comp_fold:.4f} "
            f"(GRU + Heuristic + Calib) ---"
        )

    # --- End of CV Loop ---

    # --- 4. Summarize CV Results & Determine Final Parameters ---
    logger.info("--- Step 4: Summarize CV Results & Determine Final Parameters ---")
    mean_cv_nll = np.mean(fold_nll_scores) if fold_nll_scores else np.nan
    final_heuristic_scale = (
        np.mean(fold_opt_scales)
        if fold_opt_scales
        and not np.all(np.isnan(fold_opt_scales))  # Ensure not all NaN
        else app_cfg.realization_calib.nll_calibration_initial_scale
    )
    logger.info(
        f"CV Summary (GRU+Heur+Calib): Mean CompNLL={mean_cv_nll:.4f}, "
        f"Mean Opt Scale (Final Heuristic Scale)={final_heuristic_scale:.3f}"
    )
    if fold_gru_val_loss:
        mean_gru_val_loss = np.mean(fold_gru_val_loss) if fold_gru_val_loss else np.nan
        logger.info(f"CV Summary: Mean GRU Internal Val MSE={mean_gru_val_loss:.4f}")

    # Average HPO parameters from all folds for Stage 1
    final_s1_xgb_p = stage1_ensemble.average_hpo_params(
        fold_hpo_results=fold_s1_xgb_hpo_params,
        default_params=app_cfg.model_hparams.default_xgb_params,
        model_name="XGBoost",
        seed=app_cfg.general.seed,
    )
    final_s1_rf_p = stage1_ensemble.average_hpo_params(
        fold_hpo_results=fold_s1_rf_hpo_params,
        default_params=app_cfg.model_hparams.default_rf_params,
        model_name="RandomForest",
        seed=app_cfg.general.seed,
    )
    final_s1_optimized_params: constants.OptimizedModelParams = {  # type: ignore
        "xgb_params": final_s1_xgb_p,
        "rf_params": final_s1_rf_p,
    }  # type: ignore
    logger.info(f"Final Avg S1 XGB Params: {final_s1_xgb_p}")
    logger.info(f"Final Avg S1 RF Params: {final_s1_rf_p}")

    # --- 5. Train Final Models on Full Data ---
    logger.info("--- Step 5: Train Final Models on Full Data ---")
    # Use X_full_train_enhanced and y_full_train which might be filtered by missingness step

    # --- 5a. Final Stage 1 Model Training (XGB/RF) ---
    logger.info("Training final Stage 1 (XGB/RF) models on all training data...")
    # Use fold_idx = n_folds for a unique seed for the final model, or a fixed one
    final_s1_models, final_s1_scaler, final_s1_key_out_indices = (
        stage1_ensemble.train_ensemble_models_for_fold(
            X_train_fold=X_full_train_enhanced,  # Full training data (potentially filtered)
            y_train_fold=y_full_train,
            num_derived_features=num_derived_actual,
            opt_params=final_s1_optimized_params,
            app_config=app_cfg,
            fold_idx=app_cfg.general.n_folds,  # Use n_folds as a distinct 'fold_idx' for final model seed
        )
    )

    # --- 5b. Generate R0 (Base Mean Prediction) for Final GRU Training ---
    logger.info("Generating R0 for final GRU training...")
    R0_full_train_s1 = stage1_ensemble.predict_with_ensemble(
        X_input=X_full_train_enhanced,
        models_dict=final_s1_models,
        scaler=final_s1_scaler,
        key_out_indices_0based=final_s1_key_out_indices,
        num_derived_features=num_derived_actual,  # from training
        total_num_outputs=app_cfg.columns.n_outputs,
    )
    R0_full_train_gru_in = np.expand_dims(R0_full_train_s1, axis=-1)

    # --- 5c. Final Stage 2 GRU Refiner Training ---
    logger.info(
        "Training final Stage 2 (Deterministic GRU refiner) on all R0 training data..."
    )
    final_gru_model = gru_refiner.build_deterministic_gru_refiner(
        sequence_length=app_cfg.columns.n_outputs,
        num_features_in=1,
        num_features_out=1,
        model_hparams=app_cfg.model_hparams,
    )
    gru_refiner.train_deterministic_gru_refiner(
        model=final_gru_model,
        X_train_r0=R0_full_train_gru_in,
        y_train_true=y_full_train,
        X_val_r0=None,
        y_val_true=None,  # No validation data for final GRU training
        model_hparams=app_cfg.model_hparams,
        epochs_override=app_cfg.model_hparams.final_gru_epochs_deterministic,  # Override epochs
    )

    # --- 6. Visualize on Last Validation Fold (if available) ---
    logger.info("--- Step 6: Visualize on Last Validation Fold ---")
    # Re-analyze variance structure on full training data for final visualization consistency
    final_var_analysis_params = evaluation.analyze_variance_structure(
        X_train_fold_enhanced=X_full_train_enhanced,
        y_train_fold=y_full_train,
        num_derived_features=num_derived_actual,
        app_config=app_cfg,
    )

    # Plot variance analysis results using the parameters from the full training data
    visualization.plot_variance_analysis(
        analysis_results=final_var_analysis_params, app_config=app_cfg
    )

    if X_val_last is not None and y_val_last is not None and X_val_last.size > 0:
        logger.info("Visualizing predictions on the last validation fold data...")
        # Stage 1 prediction on X_val_last (using final models)
        R0_vis_s1 = stage1_ensemble.predict_with_ensemble(
            X_input=X_val_last,
            models_dict=final_s1_models,
            scaler=final_s1_scaler,
            key_out_indices_0based=final_s1_key_out_indices,
            num_derived_features=num_derived_actual,
            total_num_outputs=app_cfg.columns.n_outputs,
        )
        R0_vis_s1_gru_in = np.expand_dims(R0_vis_s1, axis=-1)
        # Stage 2 GRU refinement (using final model)
        vis_r0_refined_squeezed = gru_refiner.predict_with_deterministic_gru_refiner(
            model=final_gru_model,
            X_r0_sequences=R0_vis_s1_gru_in,
            batch_size=app_cfg.model_hparams.gru_batch_size,
        )

        # Generate heuristic realizations for visualization using *final* variance params
        # generate_calibrated_realizations returns UNCALIBRATED realizations
        vis_reals_heuristic_unscaled = realization_generation.generate_calibrated_realizations(
            X_full_features=X_val_last,
            base_preds=vis_r0_refined_squeezed,
            var_analysis_params=final_var_analysis_params,  # Use params from full train data analysis
            num_derived_features=num_derived_actual,
            app_config=app_cfg,  # Needed for config access in worker
            master_seed_offset=app_cfg.general.n_folds
            + 1,  # Distinct seed offset for visualization
        )
        # Apply the final determined heuristic scale (found during CV) for visualization
        vis_reals_calibrated_final_scale = vis_reals_heuristic_unscaled.copy()
        base_pred_vis = vis_reals_calibrated_final_scale[
            :, 0, :
        ].copy()  # Realization 0 is base
        for r_idx in range(
            1, vis_reals_calibrated_final_scale.shape[1]
        ):  # For realizations 1 to N-1
            deviation = vis_reals_heuristic_unscaled[:, r_idx, :] - base_pred_vis
            vis_reals_calibrated_final_scale[:, r_idx, :] = base_pred_vis + (
                deviation * final_heuristic_scale
            )

        visualization.visualize_validation_examples(
            X_validation_samples_enhanced=X_val_last,
            y_validation_true=y_val_last,
            validation_realizations=vis_reals_calibrated_final_scale,
            num_derived_features=num_derived_actual,
            app_config=app_cfg,  # Needed for column info, paths for saving
            num_examples_to_display=3,  # Configurable number of examples
        )
    else:
        logger.info(
            "No validation data from last fold to visualize, or X_val_last was empty."
        )

    # --- 7. Process Test Data and Generate Submission File ---
    logger.info("--- Step 7: Process Test Data and Generate Submission ---")
    if test_df is not None and test_ids is not None:
        logger.info("Processing test data for final submission...")
        # Preprocess test data (use existing fitted imputer and potentially reduced columns)
        X_test_enhanced, _, num_derived_test = (
            preprocessing.prepare_features_and_target(
                df=test_df,
                imputer=knn_imputer_main,  # Use imputer fitted on training data (Step 2)
                app_config=app_cfg,  # Uses updated app_cfg.columns.input_cols
                is_fitting_imputer=False,  # Do NOT re-fit imputer on test data
            )
        )

        if X_test_enhanced.shape[1] != X_full_train_enhanced.shape[1]:
            # This check is crucial after potential column exclusion
            logger.error(
                f"Test data feature shape mismatch after preprocessing. "
                f"Expected {X_full_train_enhanced.shape[1]} features (from training), "
                f"got {X_test_enhanced.shape[1]}. Skipping submission generation."
            )
        elif (
            num_derived_test != num_derived_actual
        ):  # Also check derived count consistency
            logger.error(
                f"Test data derived feature count mismatch. "
                f"Expected {num_derived_actual}, got {num_derived_test}. Skipping submission."
            )
        else:
            # Stage 1 prediction on test data (using final models)
            R0_test_s1 = stage1_ensemble.predict_with_ensemble(
                X_input=X_test_enhanced,
                models_dict=final_s1_models,
                scaler=final_s1_scaler,
                key_out_indices_0based=final_s1_key_out_indices,
                num_derived_features=num_derived_actual,
                total_num_outputs=app_cfg.columns.n_outputs,
            )
            R0_test_s1_gru_in = np.expand_dims(R0_test_s1, axis=-1)
            # Stage 2 GRU refinement on test data (using final model)
            R0_test_refined_squeezed = (
                gru_refiner.predict_with_deterministic_gru_refiner(
                    model=final_gru_model,
                    X_r0_sequences=R0_test_s1_gru_in,
                    batch_size=app_cfg.model_hparams.gru_batch_size,
                )
            )

            # Generate heuristic realizations for test data using *final* variance params
            # generate_calibrated_realizations returns UNCALIBRATED realizations
            test_reals_heuristic_unscaled = realization_generation.generate_calibrated_realizations(
                X_full_features=X_test_enhanced,
                base_preds=R0_test_refined_squeezed,
                var_analysis_params=final_var_analysis_params,  # Use params from full train data analysis
                num_derived_features=num_derived_actual,
                app_config=app_cfg,  # Needed for config access in worker
                master_seed_offset=app_cfg.general.n_folds
                + 2,  # Distinct seed for test generation
            )
            # Apply final heuristic scale (found during CV) to test realizations
            test_reals_final_scale = test_reals_heuristic_unscaled.copy()
            base_pred_test = test_reals_final_scale[
                :, 0, :
            ].copy()  # Realization 0 is base
            for r_idx in range(
                1, test_reals_final_scale.shape[1]
            ):  # For realizations 1 to N-1
                deviation = test_reals_heuristic_unscaled[:, r_idx, :] - base_pred_test
                test_reals_final_scale[:, r_idx, :] = base_pred_test + (
                    deviation * final_heuristic_scale
                )

            # Prepare and save submission file
            data_io.prepare_submission_file(
                geology_sample_ids=test_ids,
                final_realizations_for_submission=test_reals_final_scale,
                submission_file_path=app_cfg.paths.submission_file,
                geology_id_col_name=app_cfg.columns.geology_id_col,
            )
    else:
        logger.warning(
            "Test data not available (test_df or test_ids is None). Skipping submission generation."
        )

    logger.info(
        f"Pipeline ({app_cfg.paths.script_version_tag}) finished successfully. "
        f"Mean CV CompNLL: {mean_cv_nll:.4f}. Final Heuristic Realization Scale: {final_heuristic_scale:.3f}"
    )


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Load application configuration
    app_config_instance = AppConfig()

    # Setup logging using the pattern from the now fully initialized PathConfig
    setup_logging(
        log_file_pattern_template=str(app_config_instance.paths.log_file_pattern),
        script_version_tag=app_config_instance.paths.script_version_tag,
    )

    # Log configuration details (excluding sensitive/non-serializable parts)
    try:
        config_dump_for_log = app_config_instance.model_dump(
            exclude={"model_hparams": {"xgb_param_space", "rf_param_space"}}
        )
        logger.info(
            f"Configuration loaded (search spaces excluded from this log dump):\n"
            f"{json.dumps(config_dump_for_log, indent=2, default=str)}"
        )
    except Exception as e_dump:
        logger.warning(
            f"Could not dump full config for logging: {e_dump}. "
            f"Proceeding with basic representation."
        )
        logger.info(
            f"AppConfig loaded (basic representation): "
            f"{repr(app_config_instance)[:1000]}..."
        )

    # Set all random seeds BEFORE calling functions that rely on them
    seed_everything(app_config_instance.general.seed)

    # --- Profiling Setup (Optional - uncomment to use) ---
    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # logger.info("cProfile profiler enabled.")

    try:
        start_pipeline_time = time.perf_counter()
        # Execute the main pipeline
        main_pipeline(app_config_instance)
        end_pipeline_time = time.perf_counter()
        logger.info(
            f"main_pipeline execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds."
        )
    except Exception as e:  # Catch any unhandled exceptions from the pipeline
        logger.exception(
            f"CRITICAL ERROR in main_pipeline execution: {e}"
        )  # Log full traceback
        sys.exit(1)  # Exit with an error code
    # finally:
    #     # --- Profiling Teardown (Optional) ---
    #     profiler.disable()
    #     stats = pstats.Stats(profiler).sort_stats('cumtime') # 'tottime' for self time
    #     # You can change 'cumtime' or 'tottime' to focus on different aspects
    #     stats.print_stats(50) # Print top 50 offenders
    #     profile_path = (
    #         app_config_instance.paths.output_dir /
    #         f"pipeline_profile_{app_config_instance.paths.script_version_tag}.prof"
    #     )
    #     stats.dump_stats(profile_path)
    #     logger.info(f"Profiling data saved to {profile_path}. "
    #                 "Use 'snakeviz {profile_path}' to view interactively.")
