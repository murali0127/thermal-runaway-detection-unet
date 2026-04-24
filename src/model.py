import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

def conv_block(inputs: tf.Tensor, num_filters: int) -> tf.Tensor:
    """
    Standard double-convolution block used in both encoder and decoder.

    Structure: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU

    Args:
        inputs      : Input feature map tensor.
        num_filters : Number of convolutional filters.

    Returns:
        Output feature map after two conv-bn-relu passes.
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(inputs: tf.Tensor, num_filters: int):
    """
    Encoder block: conv_block followed by 2×2 MaxPooling.

    The conv output (x) is retained as the skip connection feature map.
    The pooled output (p) is passed to the next encoder stage.

    Args:
        inputs      : Input tensor.
        num_filters : Number of convolutional filters.

    Returns:
        Tuple (skip_features, pooled_output):
            skip_features — full-resolution feature map for skip connection.
            pooled_output — spatially downsampled output for next stage.
    """
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs: tf.Tensor, skip_features: tf.Tensor,
                  num_filters: int) -> tf.Tensor:
    """
    Decoder block: TransposedConv (upsample) → Concatenate skip → conv_block.

    The skip connection reintroduces high-resolution spatial detail that was
    lost during encoder downsampling, enabling accurate hotspot boundary
    reconstruction.

    Args:
        inputs        : Low-resolution input from the previous decoder stage
                        or bottleneck.
        skip_features : Matching encoder feature map injected via skip
                        connection.
        num_filters   : Number of convolutional filters.

    Returns:
        Upsampled, skip-fused feature map.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2,
                               padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Full U-Net
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(input_shape: tuple = (128, 128, 1)) -> keras.Model:
    """
    Construct the complete U-Net model.

    Encoder stages and their spatial dimensions (128×128 input):
        Block 1 : 64  filters  — 128×128 → 64×64
        Block 2 : 128 filters  —  64×64  → 32×32
        Block 3 : 256 filters  —  32×32  → 16×16
        Block 4 : 512 filters  —  16×16  →  8×8

    Bottleneck:
        1024 filters at 8×8

    Decoder stages (mirror of encoder, with skip connections):
        Block 1 : 512 filters  —   8×8  → 16×16
        Block 2 : 256 filters  —  16×16 → 32×32
        Block 3 : 128 filters  —  32×32 → 64×64
        Block 4 :  64 filters  —  64×64 → 128×128

    Output: Conv2D(1, 1×1, sigmoid) → binary segmentation mask

    Args:
        input_shape : (H, W, C) — default (128, 128, 1) for grayscale thermal
                      images.

    Returns:
        Compiled Keras Model instance.
    """
    inputs = layers.Input(input_shape, name="thermal_input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    s1, p1 = encoder_block(inputs, 64)    # skip: 128×128×64,  pool: 64×64
    s2, p2 = encoder_block(p1,    128)    # skip:  64×64×128,  pool: 32×32
    s3, p3 = encoder_block(p2,    256)    # skip:  32×32×256,  pool: 16×16
    s4, p4 = encoder_block(p3,    512)    # skip:  16×16×512,  pool:  8×8

    # ── Bottleneck ────────────────────────────────────────────────────────────
    b1 = conv_block(p4, 1024)             # 8×8×1024

    # ── Decoder ──────────────────────────────────────────────────────────────
    d1 = decoder_block(b1, s4, 512)       # 16×16×512
    d2 = decoder_block(d1, s3, 256)       # 32×32×256
    d3 = decoder_block(d2, s2, 128)       # 64×64×128
    d4 = decoder_block(d3, s1,  64)       # 128×128×64

    # ── Output ────────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="segmentation_mask"
    )(d4)

    model = models.Model(inputs, outputs, name="UNet_ThermalRunaway")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_unet(input_shape=(128, 128, 1))
    model.summary()

    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"\n✓ Total trainable parameters: {trainable_params:,}")
    print(f"✓ Input  shape : {model.input_shape}")
    print(f"✓ Output shape : {model.output_shape}")
