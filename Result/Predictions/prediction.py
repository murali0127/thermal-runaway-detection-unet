
def visualize_predictions(model, test_generator, num_samples=6):
    """Visualize model predictions on test samples"""
    # Get random samples
    X_batch, y_batch = test_generator[0]
    predictions = model.predict(X_batch[:num_samples])

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original thermal image
        axes[i, 0].imshow(X_batch[i, :, :, 0], cmap='hot')
        axes[i, 0].set_title('Thermal Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(y_batch[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        # Predicted mask
        pred_mask = (predictions[i, :, :, 0] > 0.5).astype(np.float32)
        axes[i, 2].imshow(pred_mask, cmap='gray')

        # Calculate metrics for this sample
        dice = dice_coefficient(y_batch[i:i+1], predictions[i:i+1]).numpy()
        iou = iou_score(y_batch[i:i+1], predictions[i:i+1]).numpy()

        axes[i, 2].set_title(f'Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}',
                            fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{config.BASE_PATH}/test_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize predictions
visualize_predictions(best_model, test_generator, num_samples=6)
print("✓ Test predictions visualized and saved!")

