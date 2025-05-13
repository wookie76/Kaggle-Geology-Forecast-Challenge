"""
Plotting utilities for visualizing model performance, data characteristics,
and validation examples.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.ticker import MaxNLocator

# Configuration imports
from config import AppConfig, ColumnConfig, PathConfig


def visualize_validation_examples(
    X_validation_samples_enhanced: np.ndarray,
    y_validation_true: np.ndarray,
    validation_realizations: np.ndarray,
    num_derived_features: int,  # This comes from the main pipeline based on actual FE
    app_config: AppConfig,
    num_examples_to_display: int = 3,
) -> None:
    """Visualizes validation examples with input, true output, and realizations."""
    col_cfg: ColumnConfig = app_config.columns  # Original full column config
    paths_cfg: PathConfig = app_config.paths

    logger.info(
        f"Visualizing {num_examples_to_display} validation examples with realizations..."
    )

    n_total_features_in_enhanced_data = X_validation_samples_enhanced.shape[1]
    # Calculate n_original_features based on the *actual* X_validation_samples_enhanced
    # This already accounts for any excluded columns from main.py.
    n_original_features_actual = (
        n_total_features_in_enhanced_data - num_derived_features
    )

    if n_original_features_actual < 0:
        logger.error(
            "Calculated negative number of original features for visualization. "
            "Aborting plot."
        )
        return

    X_validation_original_inputs = X_validation_samples_enhanced[
        :,
        :n_original_features_actual,  # Slice using the actual number of original features
    ]

    num_available_samples = X_validation_original_inputs.shape[0]
    if num_available_samples == 0:
        logger.warning("No validation samples available to visualize.")
        return

    num_to_plot = min(num_examples_to_display, num_available_samples)
    if num_to_plot == 0:
        logger.info("Number of examples to display is 0. Skipping visualization.")
        return

    indices_to_plot = np.random.choice(
        num_available_samples, num_to_plot, replace=False
    )

    plt.style.use("dark_background")

    if num_to_plot == 1:
        fig, ax_single = plt.subplots(figsize=(16, 6))
        axs_flat = [ax_single]
    else:
        fig, axs_multiple = plt.subplots(
            nrows=num_to_plot, ncols=1, figsize=(16, 5 * num_to_plot)
        )
        axs_flat = axs_multiple.flatten()

    # --- CORRECTED input_time_axis generation ---
    # It should range from -(n_original_features_actual - 1) to 0
    # Example: if n_original_features_actual = 272, range is -271 to 0.
    # Example: if n_original_features_actual = 300, range is -299 to 0.
    if n_original_features_actual > 0:
        input_time_axis_start = -(n_original_features_actual - 1)
        input_time_axis = np.arange(input_time_axis_start, 0 + 1)  # Ends at 0
    else:  # Should not happen if we have original inputs
        input_time_axis = np.array([])

    if len(input_time_axis) != n_original_features_actual:
        logger.error(
            f"Mismatch in visualize_validation_examples: input_time_axis length ({len(input_time_axis)}) "
            f"!= n_original_features_actual ({n_original_features_actual}). Plotting may be incorrect."
        )
        # Fallback if logic error, to prevent crash, though plot will be wrong
        input_time_axis = np.arange(n_original_features_actual)

    # Output time axis (relative positions, 1 to 300 - this is fixed by problem spec)
    output_time_axis = np.arange(
        col_cfg.output_col_range_start, col_cfg.output_col_range_end + 1
    )

    num_realizations_plotted = validation_realizations.shape[1]

    for i, sample_idx in enumerate(indices_to_plot):
        ax = axs_flat[i]

        # Plot known input series (Original features)
        if X_validation_original_inputs[sample_idx, :].shape[0] == len(input_time_axis):
            ax.plot(
                input_time_axis,
                X_validation_original_inputs[
                    sample_idx, :
                ],  # Should now match input_time_axis
                color="deepskyblue",
                linestyle="-",
                linewidth=1.5,
                label="Known Input (Original)",
            )
        else:
            logger.warning(
                f"Sample {sample_idx}: Shape mismatch for input plot. X_orig: "
                f"{X_validation_original_inputs[sample_idx, :].shape[0]}, time_axis: {len(input_time_axis)}"
            )

        # ... (rest of the plotting for y_true and realizations remains the same)
        ax.plot(
            output_time_axis,
            y_validation_true[sample_idx, :],
            color="lime",
            linestyle="-",
            linewidth=2,
            label="True Future Output",
        )

        for r_iter in range(num_realizations_plotted):
            # ... (plot logic for realizations as before) ...
            is_base_prediction = r_iter == 0
            line_color = "orangered" if is_base_prediction else "tomato"
            line_alpha = 0.9 if is_base_prediction else 0.45
            line_width = 1.8 if is_base_prediction else 1.0
            line_style = "-" if is_base_prediction else "--"
            real_label: Optional[str] = None
            if is_base_prediction:
                real_label = "R0 (Base Prediction)"
            elif num_realizations_plotted <= 5:
                real_label = f"Realization {r_iter}"

            ax.plot(
                output_time_axis,
                validation_realizations[sample_idx, r_iter, :],
                color=line_color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=line_alpha,
                label=real_label,
            )

        ax.axvline(
            0,
            color="slategray",
            linestyle=":",
            linewidth=1.2,
            label="Prediction Horizon (t=0)",
        )
        ax.set_title(
            f"Validation Example (Sample Index: {sample_idx})",
            fontsize=13,
            color="lightgray",
        )
        ax.set_xlabel("Position / Time Step (relative)", fontsize=11, color="lightgray")
        ax.set_ylabel("Z-Deformation Value", fontsize=11, color="lightgray")
        legend = ax.legend(loc="best", fontsize=9)
        [text.set_color("lightgray") for text in legend.get_texts()]
        ax.grid(True, linestyle=":", alpha=0.4, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=9, colors="lightgray")
        for spine_pos in ["bottom", "top", "right", "left"]:
            ax.spines[spine_pos].set_color("gray")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(pad=2.5)
    plot_save_path = paths_cfg.validation_predictions_plot_file
    # ... (save plot logic)
    try:
        plot_save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_save_path, facecolor="black", dpi=150)
        logger.info(f"Validation examples visualization saved to: {plot_save_path}")
    except Exception as e_save:
        logger.error(f"Failed to save validation visualization plot: {e_save}")
    finally:
        plt.close(fig)


def plot_variance_analysis(
    analysis_results: Dict[str, Any], app_config: AppConfig
) -> None:
    """Plots the variance analysis results."""
    logger.info("Plotting variance analysis results...")
    paths_cfg = (
        app_config.paths
    )  # app_config.columns not directly needed for x-axis of variances if using raw index
    # but might be if original column names/positions were desired labels

    input_variances = analysis_results["input_variance_per_pos"]
    output_variances = analysis_results["output_variance_per_pos"]
    variance_scales_for_realization = analysis_results[
        "variance_scale_factors_realization"
    ]

    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # --- CORRECTED X-axis for Input Variances ---
    # input_variances corresponds to the *actual kept* original input features.
    # If these are the most recent N, their positions are -(N-1) to 0.
    num_actual_input_features = len(input_variances)
    if num_actual_input_features > 0:
        input_pos_range_actual = np.arange(-(num_actual_input_features - 1), 0 + 1)
    else:
        input_pos_range_actual = np.array([])

    if len(input_pos_range_actual) == num_actual_input_features:
        axs[0].plot(
            input_pos_range_actual,
            input_variances,
            "c-",
            label="Input Feat Var (per pos)",
        )
    else:  # Fallback or if original positions not important, plot against raw index
        axs[0].plot(input_variances, "c-", label="Input Feat Var (raw index)")
        logger.warning(
            f"Input variance plot x-axis generation might be misaligned "
            f"due to unexpected num_actual_input_features={num_actual_input_features} "
            f"vs len(input_pos_range_actual)={len(input_pos_range_actual)}"
        )

    # Output variances are fixed to output positions 1 to N_outputs
    output_pos_range_fixed = np.arange(
        app_config.columns.output_col_range_start,
        app_config.columns.output_col_range_end + 1,
    )
    if len(output_pos_range_fixed) == len(output_variances):
        axs[0].plot(
            output_pos_range_fixed,
            output_variances,
            "m-",
            label="Output Target Var (per pos)",
        )
    else:
        axs[0].plot(output_variances, "m-", label="Output Target Var (raw index)")
        logger.warning(
            f"Output variance plot length mismatch: range {len(output_pos_range_fixed)} "
            f"vs data {len(output_variances)}"
        )

    axs[0].set_title("Variance vs. Position (Input & Output)", color="lightgray")
    axs[0].set_xlabel("Position Index (relative)", color="lightgray")
    axs[0].set_ylabel("Variance (log scale)", color="lightgray")
    axs[0].legend(facecolor="dimgray", labelcolor="lightgray")
    axs[0].grid(True, linestyle=":", alpha=0.5, color="gray")
    axs[0].semilogy()
    axs[0].tick_params(colors="lightgray", which="both")
    for spine in axs[0].spines.values():
        spine.set_edgecolor("gray")

    # --- Variance Scale Factors plot (x-axis is fixed output positions) ---
    if len(output_pos_range_fixed) == len(variance_scales_for_realization):
        axs[1].plot(
            output_pos_range_fixed,
            variance_scales_for_realization,
            "lime",
            label="NLL-Derived Var Scale Factor",
        )
    else:
        axs[1].plot(
            variance_scales_for_realization,
            "lime",
            label="NLL-Derived Var Scale Factor (raw index)",
        )
        logger.warning(
            f"Var scale factor plot length mismatch: range {len(output_pos_range_fixed)} "
            f"vs data {len(variance_scales_for_realization)}"
        )

    axs[1].set_title(
        "Normalized Variance Scale Factors for Realizations", color="lightgray"
    )
    axs[1].set_xlabel("Output Position Index (relative)", color="lightgray")
    axs[1].set_ylabel("Normalized Scale Factor", color="lightgray")
    axs[1].legend(facecolor="dimgray", labelcolor="lightgray")
    axs[1].grid(True, linestyle=":", alpha=0.5, color="gray")
    axs[1].tick_params(colors="lightgray", which="both")
    for spine in axs[1].spines.values():
        spine.set_edgecolor("gray")

    plt.tight_layout(pad=2.0)
    save_path = paths_cfg.variance_plot_file
    # ... (save plot logic)
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, facecolor="black", dpi=150)
        logger.info(f"Variance analysis plot saved to: {save_path}")
    except Exception as e_save:
        logger.error(f"Failed to save variance analysis plot: {e_save}")
    finally:
        plt.close(fig)
