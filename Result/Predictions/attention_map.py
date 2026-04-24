def get_attention_map(model, image, encoder_layer_names):
attention_maps = []
    for layer_name in encoder_layer_names:
        try:
            # Create a model that outputs the feature map of the specified encoder layer
            feature_extractor = tf.keras.models.Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
            # Get the feature map for the input image
            feature_map = feature_extractor.predict(image, verbose=0)[0] # (H_conv, W_conv, C_conv)

            # Average across channels to get a single 2D activation map
            activation_map = np.mean(feature_map, axis=-1) # (H_conv, W_conv)

            # Normalize to [0, 1]
            activation_map = activation_map - np.min(activation_map)
            if np.max(activation_map) > 0:
                activation_map = activation_map / np.max(activation_map)

            # Resize to original image dimensions
            resized_map = cv2.resize(activation_map,
                                     (image.shape[2], image.shape[1]),
                                     interpolation=cv2.INTER_LINEAR)
            attention_maps.append(resized_map)
        except ValueError as e:
            print(f"Warning: Could not get layer {layer_name}. Error: {e}")
            attention_maps.append(None) # Append None if layer not found or error
    return attention_maps
def visualize_attention_maps(model, test_generator, encoder_layer_names, num_samples=3):
    """
    Visualizes attention maps from specified encoder layers overlaid on thermal images.
    Plots: Thermal Image | Ground Truth | Prediction | Attention Map (for each specified layer)
    """
    X_batch, y_batch = test_generator[0]
    predictions = model.predict(X_batch[:num_samples], verbose=0)

    # Determine number of columns needed based on encoder layers for attention
    num_attention_cols = len(encoder_layer_names)

    fig, axes = plt.subplots(num_samples, 3 + num_attention_cols,
                             figsize=(4 * (3 + num_attention_cols), 4 * num_samples))
    fig.suptitle('U-Net Encoder Attention Maps (Feature Map Visualizations)',
                 fontsize=16, fontweight='bold', y=1.01)

    col_titles = ['Thermal Image', 'Ground Truth Mask', 'Predicted Mask'] + [f'Attention: {name.replace('conv2d_', 's')}' for name in encoder_layer_names]

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold')

    for i in range(num_samples):
        img  = X_batch[i:i+1]           # shape (1, H, W, C)
        gt   = y_batch[i, :, :, 0]      # shape (H, W)
        pred = predictions[i, :, :, 0] # shape (H, W)

        # Get attention maps for all specified encoder layers
        attention_maps = get_attention_map(model, img, encoder_layer_names)

        # Column 1: Raw thermal image
        axes[i, 0].imshow(np.squeeze(img), cmap='hot')
        axes[i, 0].axis('off')

        # Column 2: Ground truth mask
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].axis('off')

        # Column 3: Predicted segmentation mask
        axes[i, 2].imshow(pred > 0.5, cmap='gray')
        axes[i, 2].axis('off')

        # Remaining columns: Attention maps overlaid on the thermal image
        thermal_base_img = np.squeeze(img)
        thermal_base_img_uint8 = np.uint8(thermal_base_img * 255)
        thermal_base_img_rgb = cv2.cvtColor(thermal_base_img_uint8, cv2.COLOR_GRAY2BGR)

        for j, attn_map in enumerate(attention_maps):
            if attn_map is not None:
                # Convert attention map to colormap and overlay
                attn_map_uint8 = np.uint8(255 * attn_map)
                attn_colored = cv2.applyColorMap(attn_map_uint8, cv2.COLORMAP_INFERNO)
                attn_colored = cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB)

                overlay = cv2.addWeighted(thermal_base_img_rgb, 0.4, attn_colored, 0.6, 0)
                axes[i, 3 + j].imshow(overlay)
            else:
                axes[i, 3 + j].text(0.5, 0.5, 'Map N/A',
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     transform=axes[i, 3 + j].transAxes,
                                     color='red', fontsize=12)
            axes[i, 3 + j].axis('off')

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/attention_maps_visualization.png',
                dpi=150, bbox_inches='tight')
    plt.show()

#Attention Map Visualization
# STEP 1: From the printed list in CELL XAI-3, find the 4 skip-connection
#         layer names corresponding to s1, s2, s3, s4 in your build_unet().
#         They are the last 'conv2d' layers at resolutions:
#         s1 → 128x128x64  |  s2 → 64x64x128
#         s3 → 32x32x256   |  s4 → 16x16x512
#
# STEP 2: Replace the names below with the correct ones from your model.

ENCODER_SKIP_LAYERS = [
    'conv2d_1',   # s1: output shape (None, 128, 128, 64)  — UPDATE IF NEEDED
    'conv2d_3',   # s2: output shape (None,  64,  64, 128) — UPDATE IF NEEDED
    'conv2d_5',   # s3: output shape (None,  32,  32, 256) — UPDATE IF NEEDED
    'conv2d_7',   # s4: output shape (None,  16,  16, 512) — UPDATE IF NEEDED
]

# STEP 3: Run attention map visualization on 3 test samples
visualize_attention_maps(
    model=best_model,
    test_generator=test_generator,
    encoder_layer_names=ENCODER_SKIP_LAYERS,
    num_samples=3
)
