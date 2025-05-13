"""
Pydantic models for pipeline configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import cpu_count

from pydantic import BaseModel, Field, field_validator, model_validator
from skopt.space import Integer, Real, Categorical  # For HPO param spaces

# --- Constants moved here for tighter HPO config ---
XGB_PARAM_SPACE: Dict[str, Any] = {
    "n_estimators": Integer(100, 300),
    "max_depth": Integer(3, 7),
    "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
    "subsample": Real(0.6, 1.0, prior="uniform"),
    "colsample_bytree": Real(0.6, 1.0, prior="uniform"),
    "gamma": Real(0, 5, prior="uniform"),
    "reg_alpha": Real(0, 1, prior="uniform"),
    "reg_lambda": Real(0, 1, prior="uniform"),
}

RF_PARAM_SPACE: Dict[str, Any] = {
    "n_estimators": Integer(100, 300),
    "max_depth": Integer(4, 10),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2", 0.7, 0.8, 0.9]),
}
# --- End moved HPO constants ---


class PathConfig(BaseModel):
    """Configuration for all file and directory paths."""

    base_data_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("GEOLOGY_DATA_PATH", ".")
        )  # Example default
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("GEOLOGY_OUTPUT_PATH", "./output")
        )  # Example default
    )
    script_version_tag: str = "v2_pywan_refactor"  # Updated version tag

    # These will be dynamically constructed
    train_file: Path = Path()
    test_file: Path = Path()
    submission_file: Path = Path()
    variance_plot_file: Path = Path()
    validation_predictions_plot_file: Path = Path()
    log_file_pattern: Path = Path()  # For loguru, will include {time}

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_post_init(self, __context: Any) -> None:
        """Dynamically construct file paths after basic validation."""
        self.train_file = self.base_data_path / "train.csv"
        self.test_file = self.base_data_path / "test.csv"

        # Ensure output_dir exists before creating files within it
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.submission_file = (
            self.output_dir / f"submission_pywan_{self.script_version_tag}.csv"
        )
        self.variance_plot_file = (
            self.output_dir / f"variance_analysis_pywan_{self.script_version_tag}.png"
        )
        self.validation_predictions_plot_file = (
            self.output_dir
            / f"predictions_validation_pywan_{self.script_version_tag}.png"
        )
        self.log_file_pattern = (  # Loguru will replace {time}
            self.output_dir
            / f"run_geology_forecast_{self.script_version_tag}_{{time}}.log"
        )


class GeneralConfig(BaseModel):
    """General pipeline execution parameters."""

    seed: int = 42
    n_folds: int = 5
    num_cores: int = Field(default_factory=lambda: cpu_count() or 1)
    num_realizations: int = 10  # As per competition: predict 10 realizations


class ColumnConfig(BaseModel):
    """Configuration for data columns and structure."""

    input_col_range_start: int = -299
    input_col_range_end: int = 0
    output_col_range_start: int = 1
    output_col_range_end: int = 300
    geology_id_col: str = "geology_id"

    # Dynamically populated
    input_cols: List[str] = Field(default_factory=list)
    input_length: int = 0
    output_cols: List[str] = Field(default_factory=list)
    n_outputs: int = 0
    all_numeric_feature_cols: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.input_cols = [
            str(i)
            for i in range(self.input_col_range_start, self.input_col_range_end + 1)
        ]
        self.input_length = len(self.input_cols)
        self.output_cols = [
            str(i)
            for i in range(self.output_col_range_start, self.output_col_range_end + 1)
        ]
        self.n_outputs = len(self.output_cols)
        self.all_numeric_feature_cols = self.input_cols + self.output_cols


class OriginalDerivedFeaturesConfigModel(BaseModel):
    recent_points_window: int = 15
    mvg_avg_window: int = 5
    std_dev_window: int = 15
    roc_window: int = 10
    prior_window_start: int = -15  # e.g., index -15
    prior_window_end: int = (
        -5
    )  # e.g., index -5 (exclusive for end of prior, start of recent)

    @property
    def n_features(self) -> int:
        return 6  # Based on original implementation (poly1, mean, std, poly2, roc, mean_ratio)


class WaveletConfigModel(BaseModel):
    family: str = "db4"
    decomposition_level: int = 3
    mode: str = "symmetric"
    n_features_per_level: int = 2  # e.g., std_dev and max_abs for detail coeffs
    n_features_final_approx: int = 2  # e.g., std_dev and mean for approx coeffs

    @property
    def n_derived_features(self) -> int:
        return (
            self.n_features_per_level * self.decomposition_level
        ) + self.n_features_final_approx


class ColabInspiredFeaturesConfigModel(BaseModel):
    pattern_change_window_split: int = 10
    trend_strength_window: int = 15
    spectral_window_fft: int  # This will be set to input_length

    @property
    def n_features(self) -> int:
        # trend_strength, pattern_change, spectral_balance
        return 3


class DerivedFeaturesConfig(BaseModel):
    original: OriginalDerivedFeaturesConfigModel = Field(
        default_factory=OriginalDerivedFeaturesConfigModel
    )
    wavelet: WaveletConfigModel = Field(default_factory=WaveletConfigModel)
    colab: Optional[ColabInspiredFeaturesConfigModel] = None  # Populated by custom_init
    total_n_derived_features: int = 0

    def custom_init(self, input_length: int) -> "DerivedFeaturesConfig":
        """Initializes Colab-inspired features config and total feature count."""
        self.colab = ColabInspiredFeaturesConfigModel(
            spectral_window_fft=input_length
            # pattern_change_window_split and trend_strength_window use their defaults
        )
        if self.colab:  # Should always be true after above line
            self.total_n_derived_features = (
                self.original.n_features
                + self.wavelet.n_derived_features
                + self.colab.n_features
            )
        return self


class ModelHyperparametersConfig(BaseModel):
    knn_n_neighbors: int = 5
    bayes_search_n_iter: int = 10  # For skopt
    bayes_search_cv_folds: int = 3  # For skopt's internal CV

    # Default HPO search spaces (can be overridden if needed)
    # These were moved from constants.py
    xgb_param_space: Dict[str, Any] = Field(
        default_factory=lambda: XGB_PARAM_SPACE.copy()
    )
    rf_param_space: Dict[str, Any] = Field(
        default_factory=lambda: RF_PARAM_SPACE.copy()
    )

    # Default best parameters (will be updated after HPO for final model training)
    default_xgb_params: Dict[str, Any] = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "n_jobs": 1,  # Crucial for preventing oversubscription when BayesSearch also uses n_jobs
        # random_state will be added dynamically
    }
    default_rf_params: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,  # Example, ensure consistency with RF_PARAM_SPACE
        "max_features": "sqrt",  # Example
        "n_jobs": -1,
        # random_state will be added dynamically
    }

    # GRU Hyperparameters (can be HPO'd separately if desired, or set here)
    gru_units: int = 64
    gru_layers: int = 1
    gru_dropout: float = 0.1
    gru_recurrent_dropout: float = 0.0
    gru_learning_rate: float = 1e-3
    gru_epochs: int = 30  # For CV folds
    gru_batch_size: int = 32
    final_gru_epochs_deterministic: int = Field(default=50)  # gru_epochs + 20

    def model_post_init(self, __context: Any) -> None:
        # Ensure final_gru_epochs_deterministic is updated if gru_epochs changes
        # This field is only used if you re-enable that direct +20 logic.
        # Better: just set final_gru_epochs_deterministic independently
        # self.final_gru_epochs_deterministic = self.gru_epochs + 20
        pass


class RealizationCalibrationConfig(BaseModel):
    """Parameters for NLL calibration and heuristic realization generation."""

    # From idealized covariance matrix section of Kaggle problem description
    log_slopes: Tuple[float, float, float] = (
        1.0406028049510443,
        0.0,
        7.835345062351012,
    )
    log_offsets: Tuple[float, float, float] = (
        -6.430669850650689,
        -2.1617411566043896,
        -45.24876794412965,
    )
    global_variance_factor: float = 0.8
    nll_calibration_initial_scale: float = 0.8
    nll_calibration_search_range_factor: float = 0.7
    nll_calibration_search_steps: int = 20

    input_smooth_thresh_ratio: float = 0.5  # For characterize_input_signal
    input_faulty_thresh_max_detail: float = 1.0  # For characterize_input_signal

    # Meta-guidance for heuristic realization generation based on input signal type
    meta_guidance_smooth_scale_adj: float = 0.8
    meta_guidance_faulty_fault_throw_adj: float = 1.2
    meta_guidance_faulty_noise_adj: float = 1.1


class AppConfig(BaseModel):
    """Root application configuration model."""

    paths: PathConfig = Field(default_factory=PathConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    columns: ColumnConfig = Field(default_factory=ColumnConfig)
    derived_features: DerivedFeaturesConfig = Field(
        default_factory=DerivedFeaturesConfig
    )
    model_hparams: ModelHyperparametersConfig = Field(
        default_factory=ModelHyperparametersConfig
    )
    realization_calib: RealizationCalibrationConfig = Field(
        default_factory=RealizationCalibrationConfig
    )

    @model_validator(mode="after")
    def _initialize_dependent_configs(self) -> "AppConfig":
        """Ensure dynamically constructed parts of sub-configs are initialized."""
        # PathConfig first, as other paths might depend on output_dir
        if not self.paths.train_file:  # Check if it was default constructed
            self.paths.model_post_init(None)

        if not self.columns.input_cols:  # Check if it was default constructed
            self.columns.model_post_init(None)

        # DerivedFeaturesConfig depends on ColumnConfig's input_length
        if self.derived_features.total_n_derived_features == 0:  # Check default
            self.derived_features.custom_init(self.columns.input_length)

        # ModelHyperparametersConfig default params need the general seed
        self.model_hparams.default_xgb_params["random_state"] = self.general.seed
        self.model_hparams.default_rf_params["random_state"] = self.general.seed

        # Ensure ModelHyperparams model_post_init is called if it exists for other reasons
        self.model_hparams.model_post_init(None)

        return self
