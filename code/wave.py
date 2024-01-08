import tensorflow as tf
from tensorflow.keras import layers, Sequential
from wavetf import WaveTFFactory
from tensorflow.keras.utils import plot_model


def wave(input):
    freq = WaveTFFactory.build(kernel_type='haar', dim=2, inverse=False)(input)
    N, H, W, C = freq.shape
    c = C // 4
    LL = freq[:, :, :, :c]
    LH = freq[:, :, :, c:2 * c]
    LH = tf.expand_dims(LH, axis=1)
    HL = freq[:, :, :, 2 * c:3 * c]
    HL = tf.expand_dims(HL, axis=1)
    HH = freq[:, :, :, 3 * c:]
    HH = tf.expand_dims(HH, axis=1)
    H = layers.concatenate([LH, HL,HH], axis=1)

    return LL,H

def AddLfreq1(inputs):  ##放在卷积后
    low, high = wave(inputs)
    low = layers.Conv2D(low.shape[-1]//2, kernel_size=3, strides=1,padding='same')(low)
    low = layers.BatchNormalization()(low)
    low = tf.keras.layers.LeakyReLU(alpha=0.1)(low)

    return low