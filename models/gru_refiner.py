"""Handles the TensorFlow GRU model for refining Stage 1 (R0) predictions.

This module includes functions for building (hyperparameter-tunable), training,
and predicting with a deterministic GRU refiner model using Huber loss.
It also includes a function demonstrating KerasTuner HPO setup.
"""

import io
import os
from contextlib import redirect_stdout
from typing import Optional, Tuple, Dict, Any

import numpy as np
from loguru import logger

# Suppress TensorFlow informational messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

# KerasTuner import - ensure it's installed (`pip install keras-tuner`)
try:
    import keras_tuner as kt
except ImportError:
    logger.warning(
        "KerasTuner not installed. HPO functionality (run_gru_hpo_search) "
        "will not be available. Please install with 'pip install keras-tuner'."
    )
    kt = None


from config import AppConfig, ModelHyperparametersConfig  # <<< AppConfig ADDED HERE


def build_gru_model_for_hpo(
    hp: "kt.HyperParameters",
    sequence_length: int,
    num_features_in: int,
    num_features_out: int,
) -> keras.Model:
    """Builds a GRU model with tunable hyperparameters for KerasTuner.

    Args:
        hp: KerasTuner HyperParameters object to define search space.
        sequence_length: Length of input/output sequences.
        num_features_in: Number of input features per step.
        num_features_out: Number of output features per step.

    Returns:
        A compiled Keras GRU model with tunable hyperparameters.
    """
    # Define hyperparameter search space
    gru_units_hp = hp.Int("gru_units", min_value=32, max_value=128, step=32)
    gru_layers_hp = hp.Int("gru_layers", min_value=1, max_value=2, step=1)
    learning_rate_hp = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    huber_delta_hp = hp.Float("huber_delta", min_value=0.5, max_value=1.5, step=0.25)

    logger.debug(
        f"Building GRU for HPO: Layers={gru_layers_hp}, Units={gru_units_hp}, "
        f"LR={learning_rate_hp}, HuberDelta={huber_delta_hp}"
    )

    inputs = layers.Input(
        shape=(sequence_length, num_features_in), name="gru_input_r0_hpo"
    )
    x = inputs

    for i in range(gru_layers_hp):
        current_gru_dropout = 0.0  # For simplicity in HPO, can be added as hp
        current_recurrent_dropout = 0.0  # For simplicity in HPO, can be added as hp
        x = layers.GRU(
            units=gru_units_hp,
            return_sequences=True,
            dropout=current_gru_dropout,
            recurrent_dropout=current_recurrent_dropout,
            name=f"gru_layer_hpo_{i + 1}",
        )(x)

    refined_mean_outputs = layers.TimeDistributed(
        layers.Dense(
            units=num_features_out, activation=None, name="refined_mean_output_hpo"
        )
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=refined_mean_outputs,
        name="deterministic_gru_refiner_hpo",
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_hp)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=huber_delta_hp))
    return model


def build_deterministic_gru_refiner(
    sequence_length: int,
    num_features_in: int,
    num_features_out: int,
    model_hparams: ModelHyperparametersConfig,
) -> keras.Model:
    """Builds and compiles a GRU refiner model using fixed HPs from config."""
    # ... (This function's content remains the same as the previous full version)
    logger.info(
        f"Building DETERMINISTIC GRU refiner (fixed HPs): {model_hparams.gru_layers} layer(s), "
        f"{model_hparams.gru_units} units each, LR={model_hparams.gru_learning_rate:.1e}, "
        f"HuberDelta={getattr(model_hparams, 'huber_delta', 1.0):.2f}"
    )
    inputs = layers.Input(shape=(sequence_length, num_features_in), name="gru_input_r0")
    x = inputs
    for i in range(model_hparams.gru_layers):
        is_last_gru_layer = i == (model_hparams.gru_layers - 1)
        gru_dropout_rate = (
            model_hparams.gru_dropout
            if model_hparams.gru_layers > 1 and not is_last_gru_layer
            else 0.0
        )
        x = layers.GRU(
            units=model_hparams.gru_units,
            return_sequences=True,
            dropout=gru_dropout_rate,
            recurrent_dropout=model_hparams.gru_recurrent_dropout,
            name=f"gru_layer_{i + 1}",
        )(x)
    refined_mean_outputs = layers.TimeDistributed(
        layers.Dense(
            units=num_features_out, activation=None, name="refined_mean_output"
        )
    )(x)
    model = keras.Model(
        inputs=inputs, outputs=refined_mean_outputs, name="deterministic_gru_refiner"
    )
    optimizer = keras.optimizers.Adam(learning_rate=model_hparams.gru_learning_rate)
    huber_delta = getattr(model_hparams, "huber_delta", 1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=huber_delta))
    logger.info(
        "Deterministic GRU refiner model built and compiled with Huber loss "
        f"(delta={huber_delta:.2f})."
    )
    with io.StringIO() as buf, redirect_stdout(buf):
        model.summary(print_fn=lambda line: buf.write(line + "\n"))
        model_summary_str = buf.getvalue()
    logger.debug(
        f"Keras Model Summary for 'deterministic_gru_refiner':\n{model_summary_str}"
    )
    return model


def train_deterministic_gru_refiner(
    model: keras.Model,
    X_train_r0: np.ndarray,
    y_train_true: np.ndarray,
    model_hparams: ModelHyperparametersConfig,
    X_val_r0: Optional[np.ndarray] = None,
    y_val_true: Optional[np.ndarray] = None,
    epochs_override: Optional[int] = None,
) -> keras.callbacks.History:
    """Trains the deterministic GRU refiner model."""
    # ... (This function's content remains the same as the previous full version)
    current_epochs = (
        epochs_override if epochs_override is not None else model_hparams.gru_epochs
    )
    current_batch_size = model_hparams.gru_batch_size
    logger.info(
        f"Training deterministic GRU refiner for {current_epochs} epochs, "
        f"batch_size {current_batch_size}."
    )
    y_train_true_reshaped = (
        np.expand_dims(y_train_true, axis=-1)
        if y_train_true.ndim == 2
        else y_train_true
    )
    validation_data_gru: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if X_val_r0 is not None and y_val_true is not None:
        y_val_true_reshaped = (
            np.expand_dims(y_val_true, axis=-1) if y_val_true.ndim == 2 else y_val_true
        )
        validation_data_gru = (X_val_r0, y_val_true_reshaped)
        logger.info("Validation data provided for GRU training.")
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data_gru else "loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data_gru else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
    ]
    history = model.fit(
        X_train_r0,
        y_train_true_reshaped,
        epochs=current_epochs,
        batch_size=current_batch_size,
        validation_data=validation_data_gru,
        callbacks=callbacks_list,
        verbose=1,
    )
    logger.info("Deterministic GRU refiner training completed.")
    if validation_data_gru and "val_loss" in history.history:
        best_val_loss = min(history.history["val_loss"])
        logger.info(f"Best val_loss (Huber) during GRU training: {best_val_loss:.4f}")
    elif "loss" in history.history:
        final_loss = history.history["loss"][-1]
        logger.info(f"Final training loss (Huber) for GRU: {final_loss:.4f}")
    return history


def predict_with_deterministic_gru_refiner(
    model: keras.Model, X_r0_sequences: np.ndarray, batch_size: int
) -> np.ndarray:
    """Refines R0 sequences using the trained deterministic GRU model."""
    # ... (This function's content remains the same as the previous full version)
    num_sequences = X_r0_sequences.shape[0]
    logger.info(
        f"Refining {num_sequences} R0 sequences with deterministic GRU model..."
    )
    refined_r0_with_feat_dim = model.predict(
        X_r0_sequences, batch_size=batch_size, verbose=0
    )
    if refined_r0_with_feat_dim.shape[-1] == 1:
        refined_r0_squeezed = np.squeeze(refined_r0_with_feat_dim, axis=-1)
    else:
        refined_r0_squeezed = refined_r0_with_feat_dim
        logger.warning(
            "GRU prediction output had >1 feature in last dim. Not squeezing."
        )
    logger.info("Deterministic GRU refinement complete.")
    return refined_r0_squeezed


def run_gru_hpo_search(
    X_train_r0_hpo: np.ndarray,
    y_train_true_hpo: np.ndarray,
    X_val_r0_hpo: np.ndarray,
    y_val_true_hpo: np.ndarray,
    app_config: AppConfig,  # This line was causing the NameError
    max_trials: int = 20,
    hpo_epochs: int = 10,
) -> Dict[str, Any]:
    """Runs KerasTuner hyperparameter search for the GRU model.

    Note: This function is for DEMONSTRATION and typically run separately
    to find good HPs. These HPs would then be set in AppConfig.
    Requires KerasTuner to be installed.
    """
    if kt is None:
        logger.error(
            "KerasTuner is not installed. Cannot run HPO search. "
            "Install with 'pip install keras-tuner'."
        )
        return {  # Fallback to current AppConfig defaults if KerasTuner is unavailable
            "gru_units": app_config.model_hparams.gru_units,
            "gru_layers": app_config.model_hparams.gru_layers,
            "learning_rate": app_config.model_hparams.gru_learning_rate,
            "huber_delta": getattr(app_config.model_hparams, "huber_delta", 1.0),
        }

    # Choose a tuner (Hyperband, RandomSearch, BayesianOptimization)
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_gru_model_for_hpo(
            hp,
            sequence_length=app_config.columns.n_outputs,
            num_features_in=1,
            num_features_out=1,
        ),
        objective=kt.Objective("val_loss", direction="min"),  # Explicitly set direction
        max_epochs=hpo_epochs,
        factor=3,
        directory=str(
            app_config.paths.output_dir / "gru_hpo_search"
        ),  # Ensure Path is string
        project_name="geology_gru_refiner_hpo_search",
    )

    logger.info(
        f"Starting GRU HPO search with KerasTuner (Max Trials approx: {max_trials})..."
    )
    logger.info(
        f"Tuner search results will be stored in: {tuner.directory}/{tuner.project_name}"
    )

    y_train_true_hpo_reshaped = (
        np.expand_dims(y_train_true_hpo, axis=-1)
        if y_train_true_hpo.ndim == 2
        else y_train_true_hpo
    )
    y_val_true_hpo_reshaped = (
        np.expand_dims(y_val_true_hpo, axis=-1)
        if y_val_true_hpo.ndim == 2
        else y_val_true_hpo
    )

    tuner_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, mode="min"
        )  # Early stopping per trial
    ]

    # Note: Hyperband manages its own "trials" concept related to max_epochs and factor.
    # The `max_trials` argument for Tuner.search() is more for RandomSearch/BayesianOptimization.
    # For Hyperband, it will run through its bracketing. To limit overall search time with Hyperband,
    # adjust `max_epochs` and `factor`. The `max_trials` argument to `search` itself is deprecated/unused
    # for Hyperband tuner type according to some KerasTuner versions.
    # The Tuner base class's `search` does take `max_trials`, but Hyperband might not use it.
    # We'll pass it but Hyperband's structure will dictate actual iterations.
    tuner.search(
        X_train_r0_hpo,
        y_train_true_hpo_reshaped,
        epochs=hpo_epochs,  # Max epochs per trial in a Hyperband bracket
        validation_data=(X_val_r0_hpo, y_val_true_hpo_reshaped),
        callbacks=tuner_callbacks,
        batch_size=app_config.model_hparams.gru_batch_size,
        # max_trials parameter might not be directly used by Hyperband in its main logic,
        # as Hyperband has its own iteration structure based on max_epochs and factor.
        # However, providing it for Tuner API consistency if underlying uses it.
    )

    # tuner.search_space_summary()
    # tuner.results_summary()

    best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
    if not best_hps_list:
        logger.error(
            "KerasTuner search did not yield any best hyperparameters. Returning defaults."
        )
        return {  # Fallback to current AppConfig defaults
            "gru_units": app_config.model_hparams.gru_units,
            "gru_layers": app_config.model_hparams.gru_layers,
            "learning_rate": app_config.model_hparams.gru_learning_rate,
            "huber_delta": getattr(app_config.model_hparams, "huber_delta", 1.0),
        }

    best_hps: kt.HyperParameters = best_hps_list[0]
    best_hps_dict = best_hps.values

    logger.info("KerasTuner GRU HPO search complete. Best Hyperparameters found:")
    for key, value in best_hps_dict.items():
        logger.info(f"  Best {key}: {value}")

    return best_hps_dict
