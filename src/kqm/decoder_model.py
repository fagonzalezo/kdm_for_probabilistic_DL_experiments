from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling2D, Activation
import tensorflow as tf
tfkl = tf.keras.layers

def decoder_model(input_shape, base_depth, encoded_size, dataset, encoder_type="classification"):
    if dataset == "mnist" or dataset == "fashion_mnist":
        decoder = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=[encoded_size]),
            tfkl.Reshape([1, 1, encoded_size]),
            tfkl.Conv2DTranspose(2 * base_depth, 7, strides=1,
                                padding='valid', activation=tf.nn.gelu),
            tfkl.Conv2DTranspose(2 * base_depth, 5, strides=1,
                                padding='same', activation=tf.nn.gelu),
            tfkl.Conv2DTranspose(2 * base_depth, 5, strides=2,
                                padding='same', activation=tf.nn.gelu),
            tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                                padding='same', activation=tf.nn.gelu),
            tfkl.Conv2DTranspose(base_depth, 5, strides=2,
                                padding='same', activation=tf.nn.gelu),
            tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                                padding='same', activation=tf.nn.gelu),
            tfkl.Conv2D(filters=1, kernel_size=5, strides=1,
                        padding='same', activation=None),
        ])
        return decoder



    if dataset == "cifar10" :
        decoder_input = Input(shape=(encoded_size,))
        x = Dense(encoded_size)(decoder_input)
        x = Dense(input_shape[0] // 8 * input_shape[1] // 8 * 128)(decoder_input)
        x = Dropout(0.2)(x)
        x = Reshape((input_shape[0] // 8, input_shape[1] // 8, 128))(x)
        
        x = Conv2DTranspose(128, (3, 3), activation='gelu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(128, (3, 3), activation='gelu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2DTranspose(64, (3, 3), activation='gelu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, (3, 3), activation='gelu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2DTranspose(32, (3, 3), activation='gelu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (3, 3), activation='gelu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2DTranspose(input_shape[2], (3, 3), padding='same')(x)
        x = Activation('sigmoid')(x)
        
        decoder_model = Model(decoder_input, x)
        
        return decoder_model

