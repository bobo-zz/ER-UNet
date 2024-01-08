import tensorflow as tf
from tensorflow.keras import layers,losses,models,regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers,losses,models,regularizers

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)
def create_model(input_shape=(416, 448, 3), # if input(IR), input shape = 4165,448,4
                 start_neurons=16,
                 num_outputs=1,
                 activation=my_leaky_relu):

    inputs = layers.Input(shape=input_shape)
    encoder0_pool, encoder0 = encoder_block(inputs, start_neurons, activation=activation)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, start_neurons * 2, activation=activation)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, start_neurons * 4, activation=activation)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, start_neurons * 8, activation=activation)
    center = conv_block(encoder3_pool, start_neurons * 16 ,activation=activation)
    decoder3 = decoder_block(center, encoder3, start_neurons * 8,activation=activation)
    decoder2 = decoder_block(decoder3, encoder2, start_neurons * 4, activation=activation)
    decoder1 = decoder_block(decoder2, encoder1, start_neurons * 2, activation=activation)
    decoder0 = decoder_block(decoder1, encoder0, start_neurons, activation=activation)
    outputs = layers.Conv2D(num_outputs, (1, 1), padding='same',
                            activation='relu', name='output_layer2')(decoder0)
    outputs = layers.Activation('linear', dtype='float32')(outputs)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='model')



def conv_block(input_tensor, num_filters, activation):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    return encoder

def encoder_block(input_tensor, num_filters, activation):
    encoder = conv_block(input_tensor, num_filters, activation)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters, activation=None,
                 ):
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
    m = create_model()
    m.summary()
    print(2333)
    plot_model(m, to_file=r'D:\Unet\result_529\unet.png', show_shapes=True)
