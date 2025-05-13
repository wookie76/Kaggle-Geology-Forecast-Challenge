"""
Input/Output utilities for data loading, quality checks, and submission file generation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

# Configuration import - for column names, paths etc. if needed, but primarily for types

# Global Rich Console for table outputs (can be passed around or instantiated per function)
RICH_CONSOLE = Console()


def log_data_quality_summary(
    df: pd.DataFrame, df_name: str, relevant_cols: List[str]
) -> None:
    """Logs a summary of missing values in relevant columns of a DataFrame."""
    logger.info(f"Data quality report for: {df_name}")

    actual_cols_to_check = [col for col in relevant_cols if col in df.columns]
    if not actual_cols_to_check:
        logger.warning(
            f"No relevant columns specified or found for quality summary in {df_name}."
        )
        return

    if df.empty:
        logger.warning(f"Empty DataFrame provided for quality report: {df_name}.")
        return

    missing_info = df[actual_cols_to_check].isnull().sum()
    total_rows = len(df)

    if total_rows == 0:  # Should be caught by df.empty, but defensive check
        logger.warning(f"DataFrame has 0 rows, skipping quality report for {df_name}.")
        return

    table = Table(
        title=f"Missing Value Report for {df_name} (Relevant Columns)", show_lines=True
    )
    table.add_column("Column", style="cyan", no_wrap=True, min_width=20)
    table.add_column("Missing Count", style="magenta", justify="right")
    table.add_column("Missing %", style="green", justify="right")

    has_missing_data = False
    for col_name in actual_cols_to_check:
        missing_count = missing_info[col_name]
        if missing_count > 0:
            has_missing_data = True
            missing_percentage = (missing_count / total_rows) * 100
            table.add_row(col_name, str(missing_count), f"{missing_percentage:.2f}%")

    if has_missing_data:
        RICH_CONSOLE.print(table)
    else:
        logger.info(
            f"No missing values found in checked relevant columns of {df_name}."
        )


def load_single_csv(
    file_path: Path,
    id_col_name: str,
    numeric_cols: List[
        str
    ],  # All numeric columns expected, including ID if it's numeric pre-cast
    dataset_description: str,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Loads a single CSV file, performs basic checks, and extracts IDs.
    numeric_cols should contain all columns intended to be numeric after loading.
    id_col_name will be cast to string for IDs.
    """
    logger.info(f"Loading {dataset_description} data from: {file_path}")
    if not file_path.exists():
        logger.error(
            f"{dataset_description} file not found: {file_path}. Critical error."
        )
        return None, None  # Or raise FileNotFoundError

    # Define dtypes for robust loading
    col_dtypes: Dict[str, Any] = {
        col: float for col in numeric_cols if col != id_col_name
    }
    col_dtypes[id_col_name] = (
        str  # Load ID as string initially to prevent numeric interpretation issues
    )

    # Use specific columns to load, if numeric_cols represents the exact subset needed
    # This can save memory if CSVs are very wide.
    # Ensure id_col_name is in columns to load.
    columns_to_load = list(set([id_col_name] + numeric_cols))

    try:
        df = pd.read_csv(
            file_path,
            usecols=columns_to_load,  # Load only specified numeric and ID columns
            dtype=col_dtypes,  # Apply dtypes for columns being loaded
            low_memory=False,  # Can help with mixed types if dtypes not exhaustive
        )
    except Exception as e:
        logger.error(f"Failed to load CSV {file_path}: {e}")
        return None, None

    if id_col_name not in df.columns:
        logger.error(
            f"ID column '{id_col_name}' missing in {dataset_description} file: {file_path}."
        )
        return None, None

    geology_ids = df[id_col_name].astype(str).values  # Ensure IDs are strings

    # Log quality summary for the numeric columns (excluding ID for this particular summary)
    # Pass numeric_cols loaded, not df.columns in case CSV had more than requested
    log_data_quality_summary(
        df, dataset_description, [col for col in numeric_cols if col != id_col_name]
    )

    return df, geology_ids


def prepare_submission_file(
    geology_sample_ids: np.ndarray,
    final_realizations_for_submission: np.ndarray,  # Shape: (n_samples, n_realizations, n_output_steps)
    submission_file_path: Path,
    geology_id_col_name: str,
    # column_cfg: ColumnConfig # If needed for output_col_range or names, but currently hardcoded to 1-based pos
) -> None:
    """
    Prepares and saves the submission file in the required format.
    Realization 0 is base_preds[:, 0, :], others are r_1_pos_x etc.
    """
    logger.info(f"Preparing submission file at: {submission_file_path}")

    _n_samples, num_realizations_total, num_output_steps = (
        final_realizations_for_submission.shape
    )

    # The competition implies 10 realizations (0-9). The file format shown includes
    # columns for '1' (which is R0), and then r_1_pos_x through r_9_pos_x.
    # This means num_realizations_total should be 10.

    if num_realizations_total != 10:  # As per typical Kaggle comp: 1 (base) + 9 others
        logger.warning(
            f"Expected 10 total realizations (0-9) for submission format, "
            f"but got {num_realizations_total}. Adjusting logic if possible or submission might be incorrect."
        )
        # If num_realizations_total > 10, take the first 10.
        # If < 10, it will be problematic for the submission format.
        # For now, assume it's 10.

    submission_data: Dict[str, Any] = {geology_id_col_name: geology_sample_ids}

    # Add realization 0 (columns "1" through "300")
    for i_output_step in range(num_output_steps):
        col_name = str(
            i_output_step + 1
        )  # Output positions are 1-indexed in submission
        submission_data[col_name] = final_realizations_for_submission[
            :, 0, i_output_step
        ]

    # Add realizations 1 through 9 (columns "r_1_pos_1" through "r_9_pos_300")
    # num_realizations_total should include realization 0. So we iterate from 1 to num_realizations_total-1.
    # For a total of 10 realizations (0-9), this loop is range(1, 10) giving r_idx_model values 1-9.
    for r_idx_model in range(
        1, num_realizations_total
    ):  # Iterate for realizations 1 through N-1
        for i_output_step in range(num_output_steps):
            col_name = f"r_{r_idx_model}_pos_{i_output_step + 1}"
            submission_data[col_name] = final_realizations_for_submission[
                :, r_idx_model, i_output_step
            ]

    submission_df = pd.DataFrame(submission_data)

    try:
        submission_file_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure dir exists
        submission_df.to_csv(submission_file_path, index=False)
        logger.info(f"Submission file successfully saved: {submission_file_path}")
    except Exception as e:
        logger.error(f"Failed to save submission file to {submission_file_path}: {e}")
