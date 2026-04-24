import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_gradcam_heatmap(model, image, last_conv_layer_name='conv2d_17'):
    """
    Generates a Grad-CAM heatmap for a U-Net model.

    Args:
        model           : Your trained best_model
        image           : Single image, shape (1, 128, 128, 1)
        last_conv_layer_name : Last conv layer in the bottleneck
                               (b1 = conv_block(p4, 1024) → 'conv2d_17' by default)
                               Run model.summary() to confirm the exact name.
    Returns:
        heatmap         : np.array of shape (128, 128), values in [0, 1]
    """
    # Build a sub-model: input → last conv layer + final output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,  # bottleneck features
            model.output                                    # segmentation mask
        ]
    )

    with tf.GradientTape() as tape:
        # Forward pass — watch the bottleneck feature map
        conv_outputs, predictions = grad_model(image, training=False)
        tape.watch(conv_outputs)

        # Loss = mean activation of the predicted segmentation mask
        # This tells Grad-CAM: "explain what caused the hotspot prediction"
        loss = tf.reduce_mean(predictions)

    # Compute gradient of loss w.r.t. bottleneck feature map
    grads = tape.gradient(loss, conv_outputs)          # shape: (1, H, W, C)

    # Global Average Pooling over spatial dims → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weight each feature map channel by its importance
    conv_outputs = conv_outputs[0]                         # shape: (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # shape: (H, W, 1)
    heatmap = tf.squeeze(heatmap)                          # shape: (H, W)

    # Normalize to [0, 1] — ReLU removes negative influence
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to original image size (128x128)
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    return heatmap


def overlay_gradcam(image, heatmap, alpha=0.5):
    """
    Overlays the Grad-CAM heatmap on the thermal image.

    Args:
        image   : np.array shape (128, 128, 1) or (128, 128)
        heatmap : np.array shape (128, 128), values in [0, 1]
        alpha   : Transparency blend factor (0=only image, 1=only heatmap)
    Returns:
        superimposed_img : np.array (128, 128, 3), RGB overlay
    """
    # Convert grayscale thermal image to uint8 RGB
    img = np.squeeze(image)                            # (128, 128)
    img_uint8 = np.uint8(img * 255)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    # Convert heatmap to colormap (jet = blue→green→red)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: thermal image + heatmap
    superimposed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def visualize_gradcam(model, test_generator, num_samples=4,
                      last_conv_layer_name='conv2d_17'):
    """
    Full Grad-CAM visualization pipeline.
    Plots: Thermal Image | Ground Truth | Prediction | Grad-CAM Overlay
    """
    X_batch, y_batch = test_generator[0]
    predictions = model.predict(X_batch[:num_samples], verbose=0)

    fig, axes = plt.subplots(num_samples, 4, figsize=(18, 4.5 * num_samples))
    fig.suptitle('Grad-CAM: Why did the model predict a hotspot?',
                 fontsize=16, fontweight='bold', y=1.01)

    col_titles = ['Thermal Image', 'Ground Truth Mask',
                  'Predicted Mask', 'Grad-CAM Heatmap']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold')

    for i in range(num_samples):
        img = X_batch[i:i+1]           # shape (1, 128, 128, 1)
        gt  = y_batch[i, :, :, 0]      # shape (128, 128)
        pred = predictions[i, :, :, 0] # shape (128, 128)

        # Generate Grad-CAM heatmap for this sample
        heatmap = get_gradcam_heatmap(model, img, last_conv_layer_name)
        overlay = overlay_gradcam(img[0], heatmap)

        # Column 1: Raw thermal image
        axes[i, 0].imshow(np.squeeze(img), cmap='hot')
        axes[i, 0].axis('off')

        # Column 2: Ground truth mask
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].axis('off')

        # Column 3: Predicted segmentation mask
        axes[i, 2].imshow(pred > 0.5, cmap='gray')
        axes[i, 2].axis('off')

        # Column 4: Grad-CAM overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis('off')

        # Add small text label: where model is focusing
        max_val = np.max(heatmap)
        focus_label = 'HIGH focus on hotspot' if max_val > 0.6 else 'Moderate focus'
        axes[i, 3].set_xlabel(focus_label, fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/gradcam_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()

#visualize Grad-CAM activation
# STEP 1: From the printed list in CELL K9PVhcsI6i1s above,
#         Find the Conv2D layer with output shape (None, 8, 8, 1024).
#         Replace 'conv2d_17' below with the correct name if different.

BOTTLENECK_LAYER = 'conv2d_9'   # ← UPDATED based on model summary

# STEP 2: Run Grad-CAM on 4 test samples
visualize_gradcam(
    model=best_model,
    test_generator=test_generator,
    num_samples=4,
    last_conv_layer_name=BOTTLENECK_LAYER
)
