import tensorflow as tf
from model import build_unet, focal_dice_loss
from utils import load_dataset

# Paths
TRAIN_IMG = "dataset/images"
TRAIN_MASK = "dataset/masks"

# Load data
X_train, y_train = load_dataset(TRAIN_IMG, TRAIN_MASK)

# Build model
model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_dice_loss,
    metrics=['accuracy']
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_split=0.2
)

# Save model
model.save("unet_thermal_runaway.h5")
