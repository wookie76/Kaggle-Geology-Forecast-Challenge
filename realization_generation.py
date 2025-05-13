"""Defines strategies and orchestrates generation of diverse heuristic realizations.

This module provides the `BaseRealizationStrategy` for defining how individual
realizations are generated from a base prediction, and the main orchestrator
function `generate_calibrated_realizations` which uses these strategies in
parallel for a batch of samples. The term "calibrated" here refers to the use
of fold-specific or data-subset-specific variance analysis parameters during
the heuristic generation process, not the application of a final global NLL
calibration scale factor (which is done by the caller).
"""

from abc import (
    ABC,
)  # abstractmethod not strictly used if BaseRealizationStrategy.apply not abstract
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter
from tqdm import tqdm

from config import AppConfig, RealizationCalibrationConfig
from constants import (
    REALIZATION_STRATEGIES_CONFIG_LIST,
    InputSignalType,
    TrendType,
)

# Corrected imports from the now vectorized/public API of feature_engineering
from feature_engineering import (
    calculate_series_std_dev,  # Public function for single series std
    get_valid_points_for_series,  # Public function for single series valid points
    characterize_input_signal,
)


class RealizationStrategyParams:
    """Dataclass-like container for parameters passed to strategy.apply()."""

    def __init__(
        self,
        base_std_noise: float,
        mean_autocorrelation_dist: float,
        metric_variance_scale_factors: np.ndarray,
        global_variance_factor: float,
        meta_noise_adjustment: float,
        meta_fault_adjustment: float,
        num_outputs: int,
        rng: np.random.Generator,
    ):
        self.base_std_noise = base_std_noise
        self.mean_autocorrelation_dist = mean_autocorrelation_dist
        self.metric_variance_scale_factors = metric_variance_scale_factors
        self.global_variance_factor = global_variance_factor
        self.meta_noise_adjustment = meta_noise_adjustment
        self.meta_fault_adjustment = meta_fault_adjustment
        self.num_outputs = num_outputs
        self.rng = rng


class BaseRealizationStrategy:
    """Base class for realization generation strategies."""

    def __init__(
        self,
        name: str,
        scale: float,
        corr: float,
        trend: TrendType,
        t_amp: float,
        smooth: bool,
        **kwargs: Any,
    ):
        self.name: str = name
        self.scale_base: float = scale
        self.corr_factor: float = corr
        self.trend_type: TrendType = trend
        self.trend_amp_factor: float = t_amp
        self.smooth_final_sg: bool = smooth
        self.use_laplacian: bool = kwargs.get("use_laplacian", False)
        self.use_median: bool = kwargs.get("use_median", False)
        self.median_kernel_factor: float = kwargs.get("median_kernel_factor", 0.05)
        self.is_fault: bool = kwargs.get("is_fault", False)
        self.fault_min_throw_factor: float = kwargs.get("f_min", 0.5)
        self.fault_max_throw_factor: float = kwargs.get("f_max", 2.0)
        self._current_corr_len: float = 0.0

    def apply(
        self, base_prediction_sample: np.ndarray, params: RealizationStrategyParams
    ) -> np.ndarray:
        """Applies the strategy to generate one realization."""
        realization_data = base_prediction_sample.copy()
        unit_noise = (
            params.rng.laplace(0, 1.0 / np.sqrt(2), params.num_outputs)
            if self.use_laplacian
            else params.rng.normal(0, 1.0, params.num_outputs)
        )
        noise_magnitude = (
            params.base_std_noise
            * self.scale_base
            * params.global_variance_factor
            * params.meta_noise_adjustment
        )
        scaled_noise_unfiltered = (
            unit_noise
            * params.metric_variance_scale_factors[: params.num_outputs]
            * noise_magnitude
        )
        processed_noise: np.ndarray
        if self.use_median:
            kernel_size_float = (
                params.mean_autocorrelation_dist * self.median_kernel_factor
            )
            kernel_size = max(3, int(kernel_size_float * 2) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if 0 < kernel_size < params.num_outputs:
                processed_noise = medfilt(
                    scaled_noise_unfiltered, kernel_size=kernel_size
                )
            else:
                processed_noise = scaled_noise_unfiltered
        else:
            self._current_corr_len = max(
                params.mean_autocorrelation_dist * self.corr_factor, 0.1
            )
            processed_noise = gaussian_filter1d(
                scaled_noise_unfiltered, sigma=self._current_corr_len
            )
        realization_data += processed_noise

        if self.is_fault:
            f_loc_max = params.num_outputs - 1
            f_loc = params.rng.integers(0, f_loc_max + 1) if f_loc_max >= 0 else 0
            if params.num_outputs == 1:
                f_loc = 0
            f_throw_base = params.base_std_noise * params.rng.uniform(
                self.fault_min_throw_factor, self.fault_max_throw_factor
            )
            f_throw = (
                f_throw_base * params.meta_fault_adjustment * params.rng.choice([-1, 1])
            )
            if f_loc < params.num_outputs:
                realization_data[f_loc:] += f_throw
        elif self.trend_type != TrendType.NONE:
            trend_mag = params.base_std_noise * self.trend_amp_factor
            x_lin = np.linspace(0, 1, params.num_outputs)
            trend_comp = np.zeros(params.num_outputs)
            if self.trend_type == TrendType.RISING:
                trend_comp = np.linspace(0, trend_mag, params.num_outputs)
            elif self.trend_type == TrendType.FALLING:
                trend_comp = np.linspace(trend_mag, 0, params.num_outputs)
            elif self.trend_type == TrendType.RISING_LATE:
                exp_t = np.exp(3 * x_lin)
                trend_comp = trend_mag * (exp_t - 1) / (np.exp(3) - 1 + 1e-9)
            elif self.trend_type == TrendType.FALLING_LATE:
                exp_t = np.exp(3 * x_lin)
                trend_comp = trend_mag * (1 - (exp_t - 1) / (np.exp(3) - 1 + 1e-9))
            elif self.trend_type == TrendType.OSCILLATE_SHORT:
                trend_comp = trend_mag * np.sin(
                    np.linspace(0, 4 * np.pi, params.num_outputs)
                )
            elif self.trend_type == TrendType.OSCILLATE_LONG:
                trend_comp = trend_mag * np.sin(
                    np.linspace(0, 2 * np.pi, params.num_outputs)
                )
            realization_data += trend_comp
        return realization_data

    def get_current_corr_len(self) -> float:
        return self._current_corr_len

    def requires_savgol_smoothing(self) -> bool:
        return self.smooth_final_sg


def get_initialized_realization_strategies() -> List[BaseRealizationStrategy]:
    """Instantiates and returns a list of realization strategy objects."""
    return [
        BaseRealizationStrategy(**cfg) for cfg in REALIZATION_STRATEGIES_CONFIG_LIST
    ]


def _calculate_savgol_params(
    correlation_length: float, num_output_steps: int
) -> Optional[Tuple[int, int]]:
    """Calculates window & polyorder for Savitzky-Golay filter."""
    if num_output_steps <= 1:
        return None
    sg_window_len = int(correlation_length * 2) + 1
    sg_window_len = max(5, sg_window_len)
    sg_window_len = min(sg_window_len, num_output_steps)
    if sg_window_len % 2 == 0:
        sg_window_len = max(1, sg_window_len - 1)
    if sg_window_len == 0 and num_output_steps > 0:
        sg_window_len = 1
    sg_poly_order = min(2, sg_window_len - 1 if sg_window_len > 0 else 0)
    sg_poly_order = max(0, sg_poly_order)
    if not (
        sg_window_len % 2 == 1
        and sg_poly_order < sg_window_len
        and sg_window_len <= num_output_steps
        and sg_poly_order >= 0
    ):
        return None
    return sg_window_len, sg_poly_order


RealizationWorkerArgs = Tuple[
    int,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    int,
    List[BaseRealizationStrategy],
    float,
    float,
    int,
    int,
    AppConfig,
]


def _generate_realizations_for_sample_worker(args: RealizationWorkerArgs) -> np.ndarray:
    """Worker function to generate all heuristic realizations for one sample."""
    (
        s_idx,
        X_orig_s,
        base_s,
        var_p,
        n_reals_total,
        strats_list,
        init_std_out,
        global_variance_factor_cfg,
        n_output_steps,
        master_seed,
        app_cfg_worker,
    ) = args

    sample_realizations = np.zeros((n_reals_total, n_output_steps))
    if n_output_steps > 0:
        sample_realizations[0, :] = base_s[:n_output_steps]

    input_type = characterize_input_signal(
        X_orig_s,
        app_cfg_worker.derived_features.wavelet,
        app_cfg_worker.realization_calib,
    )
    cfg_rc_worker = app_cfg_worker.realization_calib
    m_noise, m_fault = 1.0, 1.0
    if input_type == InputSignalType.SMOOTH:
        m_noise = cfg_rc_worker.meta_guidance_smooth_scale_adj
    elif input_type == InputSignalType.POTENTIALLY_DISCONTINUOUS:
        m_fault = cfg_rc_worker.meta_guidance_faulty_fault_throw_adj
        m_noise = cfg_rc_worker.meta_guidance_faulty_noise_adj

    last10_pts, _ = get_valid_points_for_series(X_orig_s[-10:])  # Corrected call
    std_last10 = (
        calculate_series_std_dev(last10_pts, 0.1) if last10_pts.size > 0 else 0.1
    )  # Corrected call
    base_std_gen = max((std_last10 + init_std_out) / 2.0, 0.01)

    var_scale_factors = var_p.get(
        "variance_scale_factors_realization", np.ones(n_output_steps)
    )
    if len(var_scale_factors) != n_output_steps:
        var_scale_factors = np.ones(n_output_steps)

    for r_idx in range(1, n_reals_total):
        strat = strats_list[(r_idx - 1) % len(strats_list)]
        rng = np.random.default_rng(master_seed + s_idx + r_idx)
        strat_params = RealizationStrategyParams(
            base_std_gen,
            var_p.get("mean_autocorrelation_dist", 5.0),
            var_scale_factors,
            global_variance_factor_cfg,
            m_noise,
            m_fault,
            n_output_steps,
            rng,
        )
        current_real = strat.apply(base_s, strat_params)
        if strat.requires_savgol_smoothing():
            corr_len = strat.get_current_corr_len()
            if corr_len > 0:
                savgol_p = _calculate_savgol_params(corr_len, n_output_steps)
                if savgol_p:
                    try:
                        current_real = savgol_filter(
                            current_real, savgol_p[0], savgol_p[1]
                        )
                    except ValueError as e:
                        logger.trace(f"Savgol S{s_idx}R{r_idx}: {e}")
        if X_orig_s.size > 0:
            valid_X, _ = get_valid_points_for_series(X_orig_s)
            if valid_X.size > 0 and current_real.size > 0:
                current_real -= current_real[0] - valid_X[-1]

        if len(current_real) != n_output_steps and n_output_steps > 0:
            padded_real = np.zeros(n_output_steps)
            copy_len = min(len(current_real), n_output_steps)
            padded_real[:copy_len] = current_real[:copy_len]
            current_real = padded_real

        fb_val = np.nanmean(base_s) if not np.all(np.isnan(base_s)) else 0.0
        sample_realizations[r_idx, :] = np.nan_to_num(current_real, nan=fb_val)
    return sample_realizations


def generate_calibrated_realizations(
    X_full_features: np.ndarray,
    base_preds: np.ndarray,
    var_analysis_params: Dict[str, Any],
    num_derived_features: int,
    app_config: AppConfig,
    master_seed_offset: int = 0,
) -> np.ndarray:
    """Orchestrates heuristic realization generation for samples."""
    cfg_gen = app_config.general
    cfg_rc = app_config.realization_calib
    cfg_cols = app_config.columns
    n_reals, n_samples, n_outputs = (
        cfg_gen.num_realizations,
        X_full_features.shape[0],
        cfg_cols.n_outputs,
    )
    logger.info(f"Generating {n_reals} realizations for {n_samples} samples...")

    n_orig_feats = X_full_features.shape[1] - num_derived_features
    if n_orig_feats < 0:
        raise ValueError(f"Derived feats > total feats.")
    X_orig_only = X_full_features[:, :n_orig_feats]

    if base_preds.shape[0] != n_samples or base_preds.shape[1] != n_outputs:
        raise ValueError("Base predictions shape mismatch.")

    strats = get_initialized_realization_strategies()
    if not strats and n_reals > 1:
        logger.error("CRITICAL: No strategies. Returning tiled base predictions.")
        return np.tile(base_preds[:, np.newaxis, :], (1, n_reals, 1))

    out_var_pos = var_analysis_params.get("output_variance_per_pos", np.array([0.01]))
    init_std_h = 0.1
    if out_var_pos.size > 0:
        mean_var = np.mean(out_var_pos[: min(20, len(out_var_pos))])
        if not np.isnan(mean_var) and mean_var > 0:
            init_std_h = np.sqrt(mean_var)
    init_std_h = max(init_std_h, 0.01)

    worker_args = [
        (
            i,
            X_orig_only[i],
            base_preds[i],
            var_analysis_params,
            n_reals,
            strats,
            init_std_h,
            cfg_rc.global_variance_factor,
            n_outputs,
            cfg_gen.seed + master_seed_offset,
            app_config,
        )
        for i in range(n_samples)
    ]
    if not worker_args:
        return np.empty((0, n_reals, n_outputs))
    logger.info(f"Dispatching {len(worker_args)} tasks to {cfg_gen.num_cores} cores.")

    results = Parallel(n_jobs=cfg_gen.num_cores, backend="loky")(
        delayed(_generate_realizations_for_sample_worker)(a)
        for a in tqdm(
            worker_args, desc="Generating Realizations", disable=n_samples < 50
        )
    )
    all_reals = np.stack(results, axis=0)
    logger.info("Heuristic realization generation complete.")
    return all_reals
