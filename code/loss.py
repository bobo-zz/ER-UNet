import tensorflow as tf
import numpy as np

def weighted_mae_loss2(y_true, y_pred):
    # weights1 = tf.where(tf.greater_equal(y_pred, y_true), 1.0, 2.0)
    # weights2= tf.where(tf.greater_equal(y_pred, 0.5), 1.0, 2.0)
    #     # weights = weights1 + weights2
    #     # weights = tf.where(tf.greater_equal(y_pred, y_true), 1.0,
    #     #                    tf.where(tf.greater_equal(y_true, 0.5),
    #     #                             .5, 2.0))
    #
    weights = tf.where(tf.greater_equal(y_pred, y_true), 1.0,
                       tf.where(tf.greater_equal(y_true, 0.5), 3.0, 2.0))   
    #print(weights)
    diff = tf.abs(y_pred - y_true)
    #print(diff)
    loss = tf.reduce_mean(weights * diff, axis=-1)
    #loss = tf.reduce_mean( diff, axis=-1)
    #print(loss)
    return loss