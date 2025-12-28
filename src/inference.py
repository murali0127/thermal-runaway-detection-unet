import tensorflow as tf
import matplotlib.pyplot as plt
from utils import load_image

# Load model
model = tf.keras.models.load_model(
    "unet_thermal_runaway.h5",
    compile=False
)

# Load image
img = load_image("sample_test.png")
pred = model.predict(img[np.newaxis, ...])[0]

# Visualize
plt.subplot(1, 2, 1)
plt.title("Input")
plt.imshow(img.squeeze(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.imshow(pred.squeeze(), cmap='hot')

plt.show()
