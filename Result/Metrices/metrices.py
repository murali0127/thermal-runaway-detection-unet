# Loss Functions and Metrics

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for evaluation"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for training"""
    return 1 - dice_coefficient(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss to handle class imbalance"""
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
    focal = weight * cross_entropy
    return tf.reduce_mean(focal)

def combined_loss(y_true, y_pred):
    """Combined Focal-Dice loss as specified in the paper"""
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1e-6):
    """Intersection over Union (IoU) metric"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

print("✓ Loss functions and metrics defined!")
print("  - Combined Focal-Dice Loss")
print("  - Dice Coefficient")
print("  - IoU Score")

