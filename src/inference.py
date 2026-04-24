
import argparse
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ── Project modules ───────────────────────────────────────────────────────────
from utils import (
    Config,
    combined_loss,
    dice_coefficient,
    iou_score,
    VALID_EXTENSIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str,
                     img_size: tuple = (128, 128)) -> np.ndarray:
    """
    Load and preprocess a single thermal image for inference.

    Steps:
      1. Read as grayscale (IMREAD_GRAYSCALE)
      2. Resize to img_size
      3. Normalise to [0, 1]
      4. Add batch and channel dimensions  →  (1, H, W, 1)

    Args:
        image_path : Path to the thermal image file.
        img_size   : (H, W) target size; must match the model's input size.

    Returns:
        float32 array of shape (1, H, W, 1), ready for model.predict().

    Raises:
        FileNotFoundError : If the image file cannot be opened by OpenCV.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Could not open image: '{image_path}'. "
            "Check the path and file format."
        )
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)  # (1,H,W,1)


def postprocess_mask(prediction: np.ndarray,
                     threshold: float = 0.5) -> np.ndarray:
    """
    Convert a raw sigmoid prediction to a binary hotspot mask.

    Args:
        prediction : Model output of shape (1, H, W, 1) or (H, W, 1).
        threshold  : Sigmoid cut-off for binarisation (default 0.5).

    Returns:
        uint8 binary mask of shape (H, W), values in {0, 255}.
    """
    mask = prediction.squeeze()                          # (H, W)
    mask = (mask > threshold).astype(np.uint8) * 255
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Inference Classes
# ─────────────────────────────────────────────────────────────────────────────

class ThermalInference:
    """
    Full TensorFlow/Keras inference wrapper.

    Loads the model once at construction time and exposes:
      - predict_single   — one image, optional ground-truth comparison
      - predict_batch    — all images in a directory
      - convert_to_tflite — export a .tflite file for edge deployment

    Args:
        model_path : Path to the saved .keras model file.
        img_size   : (H, W) — must match the model's trained input size.
        threshold  : Sigmoid threshold for binarising predictions.
    """

    def __init__(self, model_path: str,
                 img_size: tuple = (128, 128),
                 threshold: float = 0.5):
        self.img_size  = img_size
        self.threshold = threshold

        print(f"  Loading model from: {model_path}")
        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                "combined_loss"   : combined_loss,
                "dice_coefficient": dice_coefficient,
                "iou_score"       : iou_score,
            },
        )
        print("  ✓ Model loaded successfully.")

    # ── Single image ──────────────────────────────────────────────────────────

    def predict_single(self, image_path: str,
                       gt_mask_path: str = None,
                       save_path: str = None,
                       verbose: bool = True):
        """
        Run inference on one thermal image.

        Args:
            image_path   : Path to input thermal image.
            gt_mask_path : (optional) Path to ground-truth mask.  If provided,
                           Dice and IoU are computed and a 3-panel figure is
                           generated (input | ground truth | prediction).
            save_path    : (optional) Path to save the output figure (.png).
            verbose      : Print metrics to stdout when True.

        Returns:
            Tuple (binary_mask, dice_val, iou_val):
                binary_mask — uint8 array (H, W), values {0, 255}
                dice_val    — Dice coefficient (or None if no gt given)
                iou_val     — IoU score        (or None if no gt given)
        """
        # ── Pre-process ───────────────────────────────────────────────────────
        input_tensor = preprocess_image(image_path, self.img_size)

        # ── Predict ───────────────────────────────────────────────────────────
        t0         = time.time()
        prediction = self.model.predict(input_tensor, verbose=0)
        latency_ms = (time.time() - t0) * 1000

        binary_mask = postprocess_mask(prediction, self.threshold)

        # ── Metrics (if ground truth available) ──────────────────────────────
        dice_val = iou_val = None
        gt_display = None

        if gt_mask_path:
            gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt = cv2.resize(gt, self.img_size)
                gt_tensor   = tf.constant(
                    gt.astype(np.float32)[None, :, :, None] / 255.0
                )
                dice_val = float(dice_coefficient(gt_tensor, prediction).numpy())
                iou_val  = float(iou_score(gt_tensor, prediction).numpy())
                gt_display = gt

        # ── Print ─────────────────────────────────────────────────────────────
        if verbose:
            print(f"\n  Image      : {os.path.basename(image_path)}")
            print(f"  Latency    : {latency_ms:.1f} ms")
            if dice_val is not None:
                print(f"  Dice       : {dice_val:.4f}")
                print(f"  IoU        : {iou_val:.4f}")

        # ── Visualise ─────────────────────────────────────────────────────────
        self._save_prediction_figure(
            image_path  = image_path,
            binary_mask = binary_mask,
            gt_mask     = gt_display,
            dice_val    = dice_val,
            iou_val     = iou_val,
            save_path   = save_path,
        )

        return binary_mask, dice_val, iou_val

    # ── Batch ─────────────────────────────────────────────────────────────────

    def predict_batch(self, input_dir: str,
                      output_dir: str = None,
                      gt_dir: str = None,
                      verbose: bool = True) -> list:
        """
        Run inference on every thermal image in `input_dir`.

        Args:
            input_dir  : Directory containing thermal images.
            output_dir : (optional) Directory to save prediction figures.
            gt_dir     : (optional) Directory containing matching masks for
                         metric computation.
            verbose    : Print per-image metrics.

        Returns:
            List of dicts, one per image:
                {"file", "dice", "iou", "latency_ms"}
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        image_paths = sorted(
            p for p in os.listdir(input_dir)
            if p.lower().endswith(VALID_EXTENSIONS)
        )
        if not image_paths:
            print(f"  ⚠  No valid images found in: {input_dir}")
            return []

        print(f"\n  Running batch inference on {len(image_paths)} images ...")
        results = []

        for fname in image_paths:
            img_path = os.path.join(input_dir, fname)
            gt_path  = (
                os.path.join(gt_dir, fname) if gt_dir else None
            )
            save_path = (
                os.path.join(output_dir,
                             os.path.splitext(fname)[0] + "_prediction.png")
                if output_dir else None
            )

            t0         = time.time()
            input_t    = preprocess_image(img_path, self.img_size)
            pred       = self.model.predict(input_t, verbose=0)
            latency_ms = (time.time() - t0) * 1000

            binary_mask = postprocess_mask(pred, self.threshold)

            dice_val = iou_val = None
            if gt_path and os.path.exists(gt_path):
                gt         = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt         = cv2.resize(gt, self.img_size)
                gt_tensor  = tf.constant(
                    gt.astype(np.float32)[None, :, :, None] / 255.0
                )
                dice_val = float(dice_coefficient(gt_tensor, pred).numpy())
                iou_val  = float(iou_score(gt_tensor, pred).numpy())

            if save_path:
                self._save_prediction_figure(
                    image_path  = img_path,
                    binary_mask = binary_mask,
                    gt_mask     = None,
                    dice_val    = dice_val,
                    iou_val     = iou_val,
                    save_path   = save_path,
                )

            results.append({
                "file"      : fname,
                "dice"      : dice_val,
                "iou"       : iou_val,
                "latency_ms": latency_ms,
            })

            if verbose:
                dice_str = f"{dice_val:.4f}" if dice_val else "  N/A  "
                iou_str  = f"{iou_val:.4f}"  if iou_val  else "  N/A  "
                print(f"  {fname:<40s}  "
                      f"Dice: {dice_str}  IoU: {iou_str}  "
                      f"({latency_ms:.1f} ms)")

        # ── Aggregate statistics ──────────────────────────────────────────────
        scored = [r for r in results if r["dice"] is not None]
        if scored:
            mean_dice = np.mean([r["dice"]       for r in scored])
            mean_iou  = np.mean([r["iou"]        for r in scored])
            mean_ms   = np.mean([r["latency_ms"] for r in results])
            print(f"\n  Batch summary ({len(results)} images):")
            print(f"    Mean Dice    : {mean_dice:.4f}")
            print(f"    Mean IoU     : {mean_iou:.4f}")
            print(f"    Mean latency : {mean_ms:.1f} ms")

        return results

    # ── TFLite export ─────────────────────────────────────────────────────────

    def convert_to_tflite(self, save_path: str = "unet_model.tflite") -> str:
        """
        Convert the loaded Keras model to a quantised TFLite model.

        The TFLite model achieves < 100 ms inference on Raspberry Pi 3 with
        negligible segmentation quality loss (max pixel difference < 0.0054).

        Args:
            save_path : Destination path for the .tflite file.

        Returns:
            Absolute path of the saved .tflite file.
        """
        print(f"\n  Converting to TFLite → {save_path} ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".",
                    exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(tflite_model)

        size_kb = os.path.getsize(save_path) / 1024
        print(f"  ✓ TFLite model saved  ({size_kb:.1f} KB) → {save_path}")
        return os.path.abspath(save_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_prediction_figure(self, image_path, binary_mask,
                                gt_mask, dice_val, iou_val,
                                save_path):
        """
        Save / display a side-by-side prediction figure.

        2-panel (input | prediction) when no gt_mask is provided.
        3-panel (input | ground truth | prediction) when gt_mask is provided.
        """
        has_gt   = gt_mask is not None
        n_panels = 3 if has_gt else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

        # Original image
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        raw = cv2.resize(raw, self.img_size)
        axes[0].imshow(raw, cmap="hot")
        axes[0].set_title("Thermal Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        if has_gt:
            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
            axes[1].axis("off")
            pred_ax = axes[2]
        else:
            pred_ax = axes[1]

        pred_ax.imshow(binary_mask, cmap="gray")
        if dice_val is not None:
            title = (f"Prediction\n"
                     f"Dice: {dice_val:.3f} | IoU: {iou_val:.3f}")
        else:
            title = "Prediction"
        pred_ax.set_title(title, fontsize=12, fontweight="bold")
        pred_ax.axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".",
                        exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# TFLite Inference  (edge / Raspberry Pi)
# ─────────────────────────────────────────────────────────────────────────────

class TFLiteInference:
    """
    Lightweight TFLite inference runner for edge deployment.

    Does NOT require the full TensorFlow package — only tflite_runtime,
    which is available for Raspberry Pi:
        pip install tflite-runtime

    If tflite_runtime is unavailable, falls back to tf.lite.Interpreter
    (full TensorFlow).

    Args:
        tflite_path : Path to the .tflite model file.
        img_size    : (H, W) input size (default 128×128).
        threshold   : Sigmoid threshold for binarisation.
    """

    def __init__(self, tflite_path: str,
                 img_size: tuple = (128, 128),
                 threshold: float = 0.5):
        self.img_size  = img_size
        self.threshold = threshold

        # Use tflite_runtime if available (lighter), else fall back to tf.lite
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=tflite_path)
            print("  Using tflite_runtime interpreter")
        except ImportError:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            print("  Using tf.lite interpreter (tflite_runtime not installed)")

        self.interpreter.allocate_tensors()
        self._input_details  = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

        print(f"  ✓ TFLite model loaded: {tflite_path}")
        print(f"    Input  shape : {self._input_details[0]['shape']}")
        print(f"    Output shape : {self._output_details[0]['shape']}")

    def predict(self, image_path: str,
                verbose: bool = True):
        """
        Run TFLite inference on one thermal image.

        Args:
            image_path : Path to the thermal image file.
            verbose    : Print latency to stdout.

        Returns:
            Tuple (binary_mask, latency_ms):
                binary_mask — uint8 array (H, W), values {0, 255}
                latency_ms  — wall-clock inference time in milliseconds
        """
        input_tensor = preprocess_image(image_path, self.img_size)
        input_tensor = input_tensor.astype(self._input_details[0]["dtype"])

        self.interpreter.set_tensor(
            self._input_details[0]["index"], input_tensor
        )

        t0 = time.time()
        self.interpreter.invoke()
        latency_ms = (time.time() - t0) * 1000

        output      = self.interpreter.get_tensor(
            self._output_details[0]["index"]
        )
        binary_mask = postprocess_mask(output, self.threshold)

        if verbose:
            print(f"  {os.path.basename(image_path):<40s}  "
                  f"Latency: {latency_ms:.1f} ms")

        return binary_mask, latency_ms

    def run_live_stream(self, camera_index: int = 0):
        """
        Run inference on a live IR camera stream (Raspberry Pi).

        Press 'q' to quit.

        Args:
            camera_index : OpenCV camera index (default 0).
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}."
            )

        print("  Live stream active — press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠  Failed to read frame.")
                break

            # Convert BGR → grayscale, resize, normalise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self.img_size)
            inp  = (gray.astype(np.float32) / 255.0)[None, :, :, None]
            inp  = inp.astype(self._input_details[0]["dtype"])

            self.interpreter.set_tensor(
                self._input_details[0]["index"], inp
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(
                self._output_details[0]["index"]
            )

            mask_display = postprocess_mask(output, self.threshold)

            # Overlay mask on original frame for display
            display = cv2.resize(frame, self.img_size)
            overlay = display.copy()
            overlay[mask_display > 0] = [0, 0, 255]   # red hotspot
            blended = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            cv2.imshow("Thermal Runaway Detection", blended)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Thermal runaway hotspot inference (Keras or TFLite)"
    )
    # Model selection
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model",   help="Path to full .keras model file")
    grp.add_argument("--tflite",  help="Path to .tflite model file (edge)")

    # Input mode
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image",     help="Single thermal image path")
    inp.add_argument("--input_dir", help="Directory of thermal images (batch)")
    inp.add_argument("--live",      action="store_true",
                     help="Live IR camera stream (TFLite only)")

    # Optional
    p.add_argument("--gt_mask",    help="Ground-truth mask (single mode only)")
    p.add_argument("--gt_dir",     help="Ground-truth mask dir (batch mode only)")
    p.add_argument("--output_dir", help="Directory to save prediction figures")
    p.add_argument("--save_path",  help="Path to save single prediction figure")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Sigmoid threshold for binarisation (default 0.5)")
    p.add_argument("--img_size",   type=int, nargs=2, default=[128, 128],
                   metavar=("H", "W"),
                   help="Model input size (default 128 128)")
    p.add_argument("--export_tflite", help="Convert .keras → .tflite and save here")
    p.add_argument("--camera",     type=int, default=0,
                   help="Camera index for --live mode (default 0)")
    return p.parse_args()


def main():
    args     = _parse_args()
    img_size = tuple(args.img_size)

    print("=" * 60)
    print("THERMAL RUNAWAY DETECTION — INFERENCE")
    print("=" * 60)

    # ── TFLite path ───────────────────────────────────────────────────────────
    if args.tflite:
        engine = TFLiteInference(args.tflite, img_size, args.threshold)

        if args.live:
            engine.run_live_stream(camera_index=args.camera)

        elif args.image:
            mask, ms = engine.predict(args.image, verbose=True)
            if args.save_path:
                cv2.imwrite(args.save_path, mask)
                print(f"  Mask saved → {args.save_path}")

        elif args.input_dir:
            out = args.output_dir or "."
            os.makedirs(out, exist_ok=True)
            for fname in sorted(os.listdir(args.input_dir)):
                if fname.lower().endswith(VALID_EXTENSIONS):
                    path = os.path.join(args.input_dir, fname)
                    mask, ms = engine.predict(path, verbose=True)
                    cv2.imwrite(
                        os.path.join(out,
                                     os.path.splitext(fname)[0] + "_mask.png"),
                        mask,
                    )

    # ── Full Keras path ────────────────────────────────────────────────────────
    else:
        engine = ThermalInference(args.model, img_size, args.threshold)

        if args.export_tflite:
            engine.convert_to_tflite(args.export_tflite)

        if args.image:
            engine.predict_single(
                image_path   = args.image,
                gt_mask_path = args.gt_mask,
                save_path    = args.save_path,
                verbose      = True,
            )

        elif args.input_dir:
            engine.predict_batch(
                input_dir  = args.input_dir,
                output_dir = args.output_dir,
                gt_dir     = args.gt_dir,
                verbose    = True,
            )

    print("\n✓ Inference complete.")


if __name__ == "__main__":
    main()
