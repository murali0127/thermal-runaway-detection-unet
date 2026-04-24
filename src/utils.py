
import os
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Central configuration class.

    Modify BASE_PATH to point to your Google Drive root (Colab) or local
    dataset directory before running training or inference.
    """

    # ── Image parameters ─────────────────────────────────────────────────────
    IMG_HEIGHT   : int   = 128
    IMG_WIDTH    : int   = 128
    IMG_CHANNELS : int   = 1       # Grayscale thermal images

    # ── Training parameters ──────────────────────────────────────────────────
    BATCH_SIZE    : int   = 16
    EPOCHS        : int   = 35
    LEARNING_RATE : float = 5e-4

    # ── Dataset paths (change BASE_PATH to your own directory) ───────────────
    BASE_PATH          : str = "/content/drive/MyDrive"
    TRAIN_IMAGES_PATH  : str = f"{BASE_PATH}/dataset/data/train/images"
    TRAIN_MASKS_PATH   : str = f"{BASE_PATH}/dataset/data/train/masks"
    VAL_IMAGES_PATH    : str = f"{BASE_PATH}/dataset/data/val/images"
    VAL_MASKS_PATH     : str = f"{BASE_PATH}/dataset/data/val/masks"
    TEST_IMAGES_PATH   : str = f"{BASE_PATH}/dataset/data/test/images"
    TEST_MASKS_PATH    : str = f"{BASE_PATH}/dataset/data/test/masks"

    # ── Output paths ─────────────────────────────────────────────────────────
    MODEL_SAVE_PATH  : str = f"{BASE_PATH}/unet_output/unet_final.keras"
    CHECKPOINT_PATH  : str = f"{BASE_PATH}/unet_output/unet_best.keras"
    LOG_DIR          : str = f"{BASE_PATH}/unet_output/logs"
    TRAIN_LOG_CSV    : str = f"{BASE_PATH}/unet_output/training_log.csv"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor,
                     smooth: float = 1e-6) -> tf.Tensor:
    """
    Sørensen–Dice coefficient — measures overlap between prediction and ground
    truth.

    Formula:
        Dice = (2 · |P ∩ G| + ε) / (|P| + |G| + ε)

    A value of 1.0 means perfect overlap; 0.0 means no overlap.

    Args:
        y_true  : Ground-truth binary mask tensor.
        y_pred  : Predicted probability mask tensor.
        smooth  : Small constant to avoid division by zero.

    Returns:
        Scalar Dice coefficient in [0, 1].
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice loss = 1 − Dice coefficient.

    Directly optimises mask overlap; robust to severe class imbalance
    (hotspot pixels < 5 % of total).

    Args:
        y_true : Ground-truth binary mask.
        y_pred : Predicted probability mask.

    Returns:
        Scalar loss value in [0, 1].
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> tf.Tensor:
    """
    Focal loss — down-weights easy background pixels so the model focuses
    on rare, hard-to-detect hotspot pixels.

    Formula:
        FL = −α · (1 − p_t)^γ · log(p_t)

    Args:
        y_true : Ground-truth binary mask.
        y_pred : Predicted probability mask.
        alpha  : Class-balance weight (default 0.25).
        gamma  : Focusing parameter (default 2.0).

    Returns:
        Scalar focal loss value.
    """
    eps   = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    cross_entropy = -y_true * tf.math.log(y_pred)
    weight        = alpha * y_true * tf.pow((1.0 - y_pred), gamma)
    return tf.reduce_mean(weight * cross_entropy)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combined Focal-Dice loss as specified in the published paper.

    Focal loss   → forces attention on rare hotspot pixels
    Dice loss    → maximises spatial overlap with ground-truth mask

    Together they handle both class imbalance (Focal) and boundary
    accuracy (Dice).

    Args:
        y_true : Ground-truth binary mask.
        y_pred : Predicted probability mask.

    Returns:
        Sum of focal and dice losses.
    """
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor,
              smooth: float = 1e-6) -> tf.Tensor:
    """
    Intersection over Union (Jaccard index) — evaluates how well the
    predicted hotspot region matches the ground-truth region.

    Formula:
        IoU = (|P ∩ G| + ε) / (|P ∪ G| + ε)

    Args:
        y_true : Ground-truth binary mask.
        y_pred : Predicted probability mask.
        smooth : Small constant to avoid division by zero.

    Returns:
        Scalar IoU in [0, 1].
    """
    y_true_f     = tf.reshape(y_true, [-1])
    y_pred_f     = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union        = (tf.reduce_sum(y_true_f)
                    + tf.reduce_sum(y_pred_f)
                    - intersection)
    return (intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")


def load_dataset_paths(images_dir: str, masks_dir: str):
    """
    Scan two directories and return sorted, matched lists of image/mask paths.

    Only files with extensions in VALID_EXTENSIONS are included. A warning
    is printed if the image and mask counts do not match.

    Args:
        images_dir : Directory containing thermal images.
        masks_dir  : Directory containing corresponding binary masks.

    Returns:
        Tuple (image_paths, mask_paths) — sorted lists of file paths.
    """
    image_paths = sorted(
        p for p in glob(os.path.join(images_dir, "*.*"))
        if p.lower().endswith(VALID_EXTENSIONS)
    )
    mask_paths = sorted(
        p for p in glob(os.path.join(masks_dir, "*.*"))
        if p.lower().endswith(VALID_EXTENSIONS)
    )

    print(f"  Found {len(image_paths)} images and {len(mask_paths)} masks "
          f"in '{os.path.basename(images_dir)}'")

    if len(image_paths) != len(mask_paths):
        print("  ⚠  Warning: image/mask count mismatch — check your dataset!")

    return image_paths, mask_paths


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataGenerator
# ─────────────────────────────────────────────────────────────────────────────

class DataGenerator(keras.utils.Sequence):
    """
    Keras Sequence that loads, preprocesses, and optionally augments
    batches of grayscale thermal images and their binary hotspot masks.

    Preprocessing applied to every image:
      1. Read as grayscale
      2. Resize to img_size
      3. Normalise to [0, 1]

    Augmentation (training only):
      - Random rotation  (±15°)
      - Random horizontal flip
      - Random vertical flip
      - Random brightness adjustment (×0.8 – ×1.2)

    Args:
        image_paths : List of paths to thermal images.
        mask_paths  : List of paths to binary ground-truth masks.
        batch_size  : Number of samples per batch.
        img_size    : (H, W) to resize every image/mask to.
        augment     : Enable augmentation when True (use for training only).
    """

    def __init__(self, image_paths: list, mask_paths: list,
                 batch_size: int = 8,
                 img_size: tuple = (128, 128),
                 augment: bool = False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.batch_size  = batch_size
        self.img_size    = img_size
        self.augment     = augment
        self.indexes     = np.arange(len(self.image_paths))

    # ── Sequence interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of batches per epoch (ceiling division)."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int):
        """Return the batch at position `index`."""
        idx          = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_images = [self.image_paths[i] for i in idx]
        batch_masks  = [self.mask_paths[i]  for i in idx]
        return self._load_batch(batch_images, batch_masks)

    def on_epoch_end(self):
        """Shuffle sample order after every epoch."""
        np.random.shuffle(self.indexes)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_batch(self, image_paths: list, mask_paths: list):
        """
        Load, preprocess, and (optionally) augment one batch.

        Returns:
            X : float32 array of shape (B, H, W, 1) — normalised images.
            y : float32 array of shape (B, H, W, 1) — normalised masks.
        """
        n = len(image_paths)
        X = np.zeros((n, *self.img_size, 1), dtype=np.float32)
        y = np.zeros((n, *self.img_size, 1), dtype=np.float32)

        for i, (img_path, mask_path) in enumerate(zip(image_paths,
                                                       mask_paths)):
            # ── Load image ──────────────────────────────────────────────────
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ⚠  Could not load image: {img_path}")
                continue
            img = cv2.resize(img, self.img_size)

            # ── Load mask ───────────────────────────────────────────────────
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  ⚠  Could not load mask: {mask_path}")
                continue
            mask = cv2.resize(mask, self.img_size)

            # ── Augment ─────────────────────────────────────────────────────
            if self.augment:
                img, mask = self._augment(img, mask)

            # ── Normalise to [0, 1] ─────────────────────────────────────────
            X[i] = np.expand_dims(img.astype(np.float32)  / 255.0, axis=-1)
            y[i] = np.expand_dims(mask.astype(np.float32) / 255.0, axis=-1)

        return X, y

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        """
        Apply stochastic spatial and photometric augmentations.

        All spatial transforms are applied identically to both the image
        and the mask to preserve pixel-level alignment.
        """
        # Random rotation ±15°
        if np.random.rand() > 0.5:
            angle = np.random.randint(-15, 15)
            cx, cy = img.shape[1] // 2, img.shape[0] // 2
            M    = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img  = cv2.warpAffine(img,  M, (img.shape[1],  img.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)

        # Random vertical flip
        if np.random.rand() > 0.5:
            img  = cv2.flip(img,  0)
            mask = cv2.flip(mask, 0)

        # Random brightness adjustment (image only)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img    = np.clip(img * factor, 0, 255).astype(np.uint8)

        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, save_path: str = None):
    """
    Plot and optionally save the four training metric curves:
    Loss, Dice Coefficient, IoU Score, and Accuracy.

    Args:
        history   : Keras History object returned by model.fit().
        save_path : If given, save the figure to this file path (PNG).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    panels = [
        ("loss",             "val_loss",             "Model Loss",        "Loss"),
        ("dice_coefficient", "val_dice_coefficient", "Dice Coefficient",  "Dice"),
        ("iou_score",        "val_iou_score",        "IoU Score",         "IoU"),
        ("accuracy",         "val_accuracy",         "Accuracy",          "Accuracy"),
    ]

    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(history.history[train_key], label=f"Train",      linewidth=2)
        ax.plot(history.history[val_key],   label=f"Validation", linewidth=2)
        ax.set_title(title,   fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch",  fontsize=12)
        ax.set_ylabel(ylabel,   fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Training history saved → {save_path}")

    plt.show()


def visualize_predictions(model: keras.Model,
                          generator: DataGenerator,
                          num_samples: int = 6,
                          save_path: str = None,
                          threshold: float = 0.5):
    """
    Display a panel of Thermal Image | Ground Truth | Prediction for
    `num_samples` random examples from the first batch of `generator`.

    Per-sample Dice and IoU scores are shown in each prediction title.

    Args:
        model       : Trained Keras model.
        generator   : DataGenerator (test split, no augmentation).
        num_samples : Number of rows in the figure.
        save_path   : If given, save the figure to this path (PNG).
        threshold   : Sigmoid threshold for binarising predictions (default 0.5).
    """
    X_batch, y_batch = generator[0]
    n = min(num_samples, len(X_batch))
    predictions = model.predict(X_batch[:n], verbose=0)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        # Thermal image
        axes[i, 0].imshow(X_batch[i, :, :, 0], cmap="hot")
        axes[i, 0].set_title("Thermal Image",  fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(y_batch[i, :, :, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth",   fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        # Prediction
        pred_mask = (predictions[i, :, :, 0] > threshold).astype(np.float32)
        dice_val  = dice_coefficient(y_batch[i:i+1], predictions[i:i+1]).numpy()
        iou_val   = iou_score(       y_batch[i:i+1], predictions[i:i+1]).numpy()

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title(
            f"Prediction\nDice: {dice_val:.3f} | IoU: {iou_val:.3f}",
            fontsize=12, fontweight="bold"
        )
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Prediction panel saved → {save_path}")

    plt.show()


def plot_score_distributions(dice_scores: list, iou_scores: list,
                             save_path: str = None):
    """
    Plot histograms of per-sample Dice and IoU scores across the full
    test set, with mean lines.

    Args:
        dice_scores : List of per-sample Dice values.
        iou_scores  : List of per-sample IoU values.
        save_path   : If given, save the figure to this path (PNG).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scores, label, color in [
        (axes[0], dice_scores, "Dice Coefficient", "skyblue"),
        (axes[1], iou_scores,  "IoU Score",        "lightcoral"),
    ]:
        mean_val = np.mean(scores)
        ax.hist(scores, bins=20, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_val:.3f}")
        ax.set_title(f"{label} Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel(label,     fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Score distribution saved → {save_path}")

    plt.show()


def print_performance_report(config: Config,
                             history,
                             test_metrics: dict,
                             dice_scores: list,
                             iou_scores:  list,
                             trainable_params: int,
                             train_size: int,
                             val_size:   int,
                             test_size:  int,
                             save_path:  str = None) -> str:
    """
    Build, print, and optionally save a plain-text performance report.

    Args:
        config           : Config instance (used for paths / hyperparams).
        history          : Keras History object.
        test_metrics     : Dict returned by model.evaluate().
        dice_scores      : Per-sample Dice scores from full test set.
        iou_scores       : Per-sample IoU scores from full test set.
        trainable_params : Total trainable parameter count.
        train_size       : Number of training samples.
        val_size         : Number of validation samples.
        test_size        : Number of test samples.
        save_path        : If given, write report to this .txt file.

    Returns:
        The report string.
    """
    sep  = "=" * 70
    report = f"""
{sep}
U-NET THERMAL RUNAWAY DETECTION — PERFORMANCE REPORT
{sep}
Project : Thermal Management of EV Battery Pack Using U-Net CNN
Paper   : IJIREEICE Vol.14, Issue 3, March 2026
DOI     : 10.17148/IJIREEICE.2026.14385
{sep}
MODEL CONFIGURATION
{sep}
Architecture      : U-Net with Skip Connections
Input Size        : {config.IMG_HEIGHT}×{config.IMG_WIDTH}×{config.IMG_CHANNELS}
Trainable Params  : {trainable_params:,}
Loss Function     : Combined Focal-Dice Loss
Optimizer         : Adam  (lr={config.LEARNING_RATE})
Batch Size        : {config.BATCH_SIZE}
Epochs Trained    : {len(history.history["loss"])}
{sep}
DATASET SUMMARY
{sep}
Training Samples  : {train_size}
Validation Samples: {val_size}
Test Samples      : {test_size}
{sep}
TEST SET PERFORMANCE
{sep}
"""
    for name, val in test_metrics.items():
        report += f"  {name:<25s}: {val:.4f}\n"

    report += f"""
{sep}
DETAILED STATISTICS (Test Set)
{sep}
Dice Coefficient
  Mean ± Std : {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}
  Range      : [{np.min(dice_scores):.4f}, {np.max(dice_scores):.4f}]

IoU Score
  Mean ± Std : {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}
  Range      : [{np.min(iou_scores):.4f}, {np.max(iou_scores):.4f}]
{sep}
TARGET PERFORMANCE (From Paper)
{sep}
  Classification Accuracy : > 95 %
  Dice Coefficient        : > 0.85
  IoU Score               : > 0.91
  Inference Latency       : < 100 ms

  Status: {"✓ ACHIEVED" if np.mean(dice_scores) > 0.85 else "⚠  NEEDS IMPROVEMENT"}
{sep}
"""

    print(report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        print(f"  ✓ Report saved → {save_path}")

    return report
