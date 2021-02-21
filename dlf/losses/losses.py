import tensorflow as tf
from tensorflow.keras.losses import Loss

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = tf.keras.layers.flatten(y_true)
    y_pred_f = tf.keras.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


class HuberLoss(Loss):
    r"""
        The Huber loss function describes the penalty incurred by an estimation procedure f. Huber (1964) 
        See wikipedia for details here: https://en.wikipedia.org/wiki/Huber_loss
            Args:
                - threshold: delta value in the huber loss equation.
            Returns:
                - losses: the loss value calculated with huber loss algorithm.
    """

    def __init__(self, threshold=1):
        super().__init__()
        self.threshold = threshold

    def call(y_true, y_pred):

        error = y_pred - y_true  # the a in the formula for huber loss
        is_small_error = tf.abs(error) <= self.threshold   # Checking if the |a| value is smaller than the threshold or not.

        forsmallerrorloss = tf.square(error)/2   # See wikipedia for the details of huber loss
        forbigerrorloss = self.threshold*(tf.abs(error) - 0.5*(self.threshold))

        return tf.where(is_small_error, forsmallerrorloss, forbigerrorloss)
