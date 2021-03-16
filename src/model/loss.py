import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from constant import IGNORE_INDEX

def boolean_masking(y_true, y_pred):
    mask = tf.not_equal(y_true, IGNORE_INDEX)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return y_true_masked, y_pred_masked

def sparse_categorical_crossentropy_with_mask(y_true, y_pred):
    y_true_masked , y_pred_masked = boolean_masking(y_true, y_pred)
    return tf.keras.backend.mean(sparse_categorical_crossentropy(y_true_masked, y_pred_masked))