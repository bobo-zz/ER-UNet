import tensorflow as tf
from tensorflow.keras import layers,losses,models,regularizers
import  tensorflow_addons as tfa
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers,losses,models,regularizers
from  wave import AddLfreq1
import math
from tensorflow.keras.layers import LeakyReLU

def channel_att(input):
    _, width, height, channel = input.get_shape()  # (B, W, H, C)
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_mean = tf.keras.layers.Conv2D(channel // 8, 1, activation=tf.nn.relu, name='CA1_avg_' + str(channel))(
    x_mean)  # (B, 1, 1, C // r)
    x_mean = tf.keras.layers.Conv2D(channel, 1, name='CA2_avg' + str(channel))(x_mean)  # (B, 1, 1, C)
    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.keras.layers.Conv2D(channel // 8, 1, activation=tf.nn.relu, name='CA1_max' + str(channel))(x_max)
    x_max = tf.keras.layers.Conv2D(channel, 1, name='CA2_max' + str(channel))(x_max)  # (B, 1, 1, C)(x_max)
    x = tf.keras.layers.Add()([x_mean, x_max])  # (B, 1, 1, C)
    x = layers.Activation('sigmoid')(x)  # (B, 1, 1, C)
    x = tf.keras.layers.Multiply()([input, x])  # (B, W, H, C)

    return  x

def hybrid_conv(input_tensor, num_filters, activation):
    brath1 = tf.keras.layers.Conv2D(num_filters,3, strides=1,padding='same')(input_tensor)
    brath1 = tf.keras.layers.BatchNormalization()(brath1)
    brath1 = tf.keras.layers.Activation(activation)(brath1)
    #----
    brath2 = tf.keras.layers.Conv2D(num_filters,3, strides=1,padding='same')(brath1)
    brath2 = tf.keras.layers.BatchNormalization()(brath2)
    brath2 = tf.keras.layers.Activation(activation)(brath2)
    #----
    brath3 = tf.keras.layers.SeparableConv2D(num_filters, 3, strides=1, padding='same')(brath1)
    brath3 = tf.keras.layers.BatchNormalization()(brath3)
    brath3 = tf.keras.layers.Activation(activation)(brath3)

    brath4 = tf.keras.layers.Conv2D(num_filters,3, strides=1,padding='same',dilation_rate=3)(brath3)
    brath4 = tf.keras.layers.BatchNormalization()(brath4)
    brath4 = tf.keras.layers.Activation(activation)(brath4)

    out1 = tf.keras.layers.Add()([brath3, brath4])
    out = tf.keras.layers.Concatenate()([input_tensor, brath2, out1])

    return out
def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)
def create_model(input_shape=(416, 448, 3),
                 start_neurons=16,
                 num_outputs=1,
                 activation=my_leaky_relu):

    input = layers.Input(shape=input_shape)
    encoder0_pool, encoder0 = encoder_block( input,  start_neurons, activation=activation,resnet_style=False)
    encoder1_pool, encoder1 = encoder_block( encoder0_pool, start_neurons * 2, activation=activation,resnet_style=False)
    encoder2_pool, encoder2= encoder_block(encoder1_pool, start_neurons * 4, activation=activation,resnet_style=False)
    encoder3_pool, encoder3= encoder_block(encoder2_pool, start_neurons * 8, activation=activation,resnet_style=False)
    center = conv_block(encoder3_pool, start_neurons * 16, activation=activation)
    decoder3 = decoder_block(center, encoder3, start_neurons * 8,activation=activation)
    decoder2 = decoder_block(decoder3, encoder2, start_neurons * 4, activation=activation)
    decoder1 = decoder_block(decoder2, encoder1, start_neurons * 2, activation=activation)
    decoder0 = decoder_block(decoder1, encoder0, start_neurons, activation=activation)
    outputs = layers.Conv2D(num_outputs, (1, 1), padding='same',
                            activation='relu', name='output_layer2')(decoder0)
    outputs = layers.Activation('linear', dtype='float32')(outputs)
    return tf.keras.models.Model(inputs=input, outputs=outputs, name='model')


def conv_block(input_tensor, num_filters, activation):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    return encoder

def encoder_block(input_tensor, num_filters, activation=None):
    encoder = hybrid_conv(input_tensor=input_tensor,num_filters= num_filters, activation= activation)
    low = AddLfreq1(encoder)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    encoder_pool = layers.concatenate([encoder_pool, low], axis =-1)
    encoder_pool = channel_att(input=encoder_pool)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters, activation=None):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    if concat_tensor is not None:
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)

    return decoder

if __name__ == '__main__':
    print(tf.__version__)
    # print(2333)
    m = create_model()
    m.summary()
    plot_model(m, to_file=r'D:\Unet\result_529\unet4-final.png', show_shapes=True)

