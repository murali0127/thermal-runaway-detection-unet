# U-Net Architecture - Building Blocks

def conv_block(inputs, num_filters):
    """Convolutional block: 2x(Conv2D + BatchNorm + ReLU)"""
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

def encoder_block(inputs, num_filters):
    """Encoder block: Conv Block + Max Pooling"""
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Decoder block: Upsampling + Concatenation + Conv Block"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

print("✓ U-Net building blocks defined!")



# Build Complete U-Net Model

def build_unet(input_shape=(128, 128, 1)):
    """
    Build U-Net architecture for semantic segmentation

    Architecture:
    - Encoder: 4 blocks with [64, 128, 256, 512] filters
    - Bottleneck: 1024 filters
    - Decoder: 4 blocks with [512, 256, 128, 64] filters
    - Output: Single channel with sigmoid activation
    """
    inputs = layers.Input(input_shape)

    # Encoder path
    s1, p1 = encoder_block(inputs, 64)      # 128x128 -> 64x64
    s2, p2 = encoder_block(p1, 128)         # 64x64 -> 32x32
    s3, p3 = encoder_block(p2, 256)         # 32x32 -> 16x16
    s4, p4 = encoder_block(p3, 512)         # 16x16 -> 8x8

    # Bottleneck
    b1 = conv_block(p4, 1024)               # 8x8

    # Decoder path
    d1 = decoder_block(b1, s4, 512)         # 8x8 -> 16x16
    d2 = decoder_block(d1, s3, 256)         # 16x16 -> 32x32
    d3 = decoder_block(d2, s2, 128)         # 32x32 -> 64x64
    d4 = decoder_block(d3, s1, 64)          # 64x64 -> 128x128

    # Output layer
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = models.Model(inputs, outputs, name='U-Net-Thermal-Runaway')
    return model

# Build the model
model = build_unet(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))

print("✓ U-Net model built!")
print(f"\nModel Summary:")
model.summary()

# Count parameters
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n✓ Total trainable parameters: {trainable_params:,}")

