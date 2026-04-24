
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Project modules ───────────────────────────────────────────────────────────
from model import build_unet
from utils import (
    Config,
    DataGenerator,
    load_dataset_paths,
    combined_loss,
    dice_coefficient,
    iou_score,
    plot_training_history,
    visualize_predictions,
    plot_score_distributions,
    print_performance_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_output_dirs(config: Config) -> None:
    """Create all output directories if they do not already exist."""
    for path in [
        os.path.dirname(config.MODEL_SAVE_PATH),
        os.path.dirname(config.CHECKPOINT_PATH),
        config.LOG_DIR,
        os.path.dirname(config.TRAIN_LOG_CSV),
    ]:
        if path:
            os.makedirs(path, exist_ok=True)


def _build_callbacks(config: Config) -> list:
    """
    Construct the Keras callback list used during training.

    Callbacks:
        ModelCheckpoint     — saves the epoch with highest val_dice_coefficient
        ReduceLROnPlateau   — halves lr after 5 epochs of no val_loss improvement
        EarlyStopping       — halts training after 10 stagnant epochs and
                              restores best weights
        CSVLogger           — appends per-epoch metrics to a CSV file
        TensorBoard         — writes event files for TensorBoard visualisation

    Args:
        config : Config instance supplying path and hyperparameter values.

    Returns:
        List of instantiated Keras Callback objects.
    """
    return [
        keras.callbacks.ModelCheckpoint(
            filepath        = config.CHECKPOINT_PATH,
            monitor         = "val_dice_coefficient",
            mode            = "max",
            save_best_only  = True,
            save_weights_only = False,
            verbose         = 1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 5,
            min_lr   = 1e-7,
            mode     = "min",
            verbose  = 1,
        ),
        keras.callbacks.EarlyStopping(
            monitor             = "val_loss",
            patience            = 10,
            restore_best_weights = True,
            mode                = "min",
            verbose             = 1,
        ),
        keras.callbacks.CSVLogger(config.TRAIN_LOG_CSV),
        keras.callbacks.TensorBoard(log_dir=config.LOG_DIR),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full training and evaluation pipeline."""

    config = Config()

    # ── 0. Environment info ────────────────────────────────────────────────────
    print("=" * 70)
    print("U-NET THERMAL RUNAWAY DETECTION — TRAINING PIPELINE")
    print("=" * 70)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPUs available     : {tf.config.list_physical_devices('GPU')}")
    print(f"Base path          : {config.BASE_PATH}")
    print("=" * 70)

    _ensure_output_dirs(config)

    # ── 1. Load dataset paths ─────────────────────────────────────────────────
    print("\n[1/6] Loading dataset paths ...")
    train_images, train_masks = load_dataset_paths(
        config.TRAIN_IMAGES_PATH, config.TRAIN_MASKS_PATH
    )
    val_images,   val_masks   = load_dataset_paths(
        config.VAL_IMAGES_PATH,   config.VAL_MASKS_PATH
    )
    test_images,  test_masks  = load_dataset_paths(
        config.TEST_IMAGES_PATH,  config.TEST_MASKS_PATH
    )
    print(
        f"\n  Dataset summary:\n"
        f"    Training   : {len(train_images):>5} samples\n"
        f"    Validation : {len(val_images):>5} samples\n"
        f"    Test       : {len(test_images):>5} samples"
    )

    # ── 2. Create DataGenerators ──────────────────────────────────────────────
    print("\n[2/6] Creating data generators ...")
    img_size = (config.IMG_HEIGHT, config.IMG_WIDTH)

    train_gen = DataGenerator(
        train_images, train_masks,
        batch_size = config.BATCH_SIZE,
        img_size   = img_size,
        augment    = True,   # Augmentation ON for training
    )
    val_gen = DataGenerator(
        val_images, val_masks,
        batch_size = config.BATCH_SIZE,
        img_size   = img_size,
        augment    = False,  # No augmentation for validation
    )
    test_gen = DataGenerator(
        test_images, test_masks,
        batch_size = config.BATCH_SIZE,
        img_size   = img_size,
        augment    = False,  # No augmentation for test
    )
    print(
        f"  Batches — train: {len(train_gen)}  "
        f"val: {len(val_gen)}  test: {len(test_gen)}"
    )

    # ── 3. Build & compile model ──────────────────────────────────────────────
    print("\n[3/6] Building U-Net model ...")
    model = build_unet(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    )

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss      = combined_loss,
        metrics   = [dice_coefficient, iou_score, "accuracy"],
    )

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Optimizer            : Adam  (lr={config.LEARNING_RATE})")
    print(f"  Loss                 : Combined Focal-Dice")

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    print("\n[4/6] Configuring callbacks ...")
    callback_list = _build_callbacks(config)
    print(
        "  ModelCheckpoint  — monitors val_dice_coefficient\n"
        "  ReduceLROnPlateau — patience=5, factor=0.5\n"
        "  EarlyStopping    — patience=10, restore_best_weights=True\n"
        "  CSVLogger        — writing to training_log.csv\n"
        "  TensorBoard      — writing to logs/"
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("\n[5/6] Training ...")
    print("=" * 70)
    print(f"  Epochs     : {config.EPOCHS}")
    print(f"  Batch size : {config.BATCH_SIZE}")
    print(f"  LR initial : {config.LEARNING_RATE}")
    print("=" * 70)

    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = config.EPOCHS,
        callbacks       = callback_list,
        verbose         = 1,
    )

    model.save(config.MODEL_SAVE_PATH)
    print(f"\n  ✓ Final model saved → {config.MODEL_SAVE_PATH}")
    print(f"  ✓ Best checkpoint  → {config.CHECKPOINT_PATH}")

    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(
            os.path.dirname(config.MODEL_SAVE_PATH), "training_history.png"
        ),
    )

    # ── 6. Evaluate on test set ───────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    print("=" * 70)

    # Load the best checkpoint for evaluation
    best_model = keras.models.load_model(
        config.CHECKPOINT_PATH,
        custom_objects={
            "combined_loss"   : combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou_score"       : iou_score,
        },
    )

    test_results  = best_model.evaluate(test_gen, verbose=1)
    test_metrics  = dict(zip(best_model.metrics_names, test_results))

    print("\n  Test set results:")
    for name, val in test_metrics.items():
        print(f"    {name:<25s}: {val:.4f}")

    # ── 7. Per-sample analysis ─────────────────────────────────────────────────
    print("\n  Running per-sample analysis over full test set ...")
    all_images, all_gt, all_preds = [], [], []

    for i in range(len(test_gen)):
        X_b, y_b  = test_gen[i]
        pred_b    = best_model.predict(X_b, verbose=0)
        all_images.append(X_b)
        all_gt.append(y_b)
        all_preds.append(pred_b)

    all_images = np.concatenate(all_images, axis=0)
    all_gt     = np.concatenate(all_gt,     axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)

    dice_scores = [
        dice_coefficient(all_gt[i:i+1], all_preds[i:i+1]).numpy()
        for i in range(len(all_preds))
    ]
    iou_scores_ = [
        iou_score(all_gt[i:i+1], all_preds[i:i+1]).numpy()
        for i in range(len(all_preds))
    ]

    # ── Save output artefacts ──────────────────────────────────────────────────
    out_dir = os.path.dirname(config.MODEL_SAVE_PATH)

    visualize_predictions(
        best_model, test_gen,
        num_samples = 6,
        save_path   = os.path.join(out_dir, "test_predictions.png"),
    )

    plot_score_distributions(
        dice_scores, iou_scores_,
        save_path = os.path.join(out_dir, "performance_distribution.png"),
    )

    print_performance_report(
        config          = config,
        history         = history,
        test_metrics    = test_metrics,
        dice_scores     = dice_scores,
        iou_scores      = iou_scores_,
        trainable_params = trainable_params,
        train_size      = len(train_images),
        val_size        = len(val_images),
        test_size       = len(test_images),
        save_path       = os.path.join(out_dir, "performance_report.txt"),
    )

    print("\n" + "=" * 70)
    print("✓ Training pipeline complete.")
    print(f"  All outputs saved to: {out_dir}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
