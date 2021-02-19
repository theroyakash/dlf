import tensorflow as tf

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = tf.keras.layers.flatten(y_true)
    y_pred_f = tf.keras.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)