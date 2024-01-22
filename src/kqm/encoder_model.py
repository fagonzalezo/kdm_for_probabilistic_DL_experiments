from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
tfkl = tf.keras.layers

def encoder_model(input_shape, base_depth, encoded_size, dataset, last_dense_layer_size=None, encoder_type="classification"):
    activation = tf.nn.gelu if encoder_type == "classification" else tf.nn.tanh
    if dataset == "mnist" or dataset == "fashion_mnist":

        layer_list_model = [
            tfkl.InputLayer(input_shape=input_shape),
            tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            tfkl.Conv2D(base_depth, 5, strides=1,
                        padding='same', activation=tf.nn.gelu),
            tfkl.Conv2D(base_depth, 5, strides=2,
                        padding='same', activation=tf.nn.gelu),
            tfkl.Conv2D(2 * base_depth, 5, strides=1,
                        padding='same', activation=tf.nn.gelu),
            tfkl.Conv2D(2 * base_depth, 5, strides=2,
                        padding='same', activation=tf.nn.gelu),
            tfkl.Conv2D(4 * encoded_size, 7, strides=1,
                        padding='valid', activation=tf.nn.gelu),
            tfkl.Flatten(),
            tfkl.Dropout(0.2),
            tfkl.Dense(encoded_size, activation=activation),
            tfkl.Dropout(0.2)
        ]
        if last_dense_layer_size != None:
            layer_list_model.append(tfkl.Dense(last_dense_layer_size, activation=tf.nn.gelu))

        encoder = tf.keras.Sequential(layer_list_model)
        return encoder


    if dataset == "cifar10" :
        i = Input(shape=input_shape)
        x = Conv2D(base_depth, (3, 3), activation=tf.nn.gelu, padding='same')(i)
        x = BatchNormalization()(x)
        x = Conv2D(base_depth, (3, 3), activation=tf.nn.gelu, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(2 * base_depth, (3, 3), activation=tf.nn.gelu, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(2 * base_depth, (3, 3), activation=tf.nn.gelu, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(2 * base_depth, (3, 3), activation=tf.nn.gelu, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(2 * base_depth, (3, 3), activation=tf.nn.gelu, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        
        # Hidden layer
        x = Dense(encoded_size, activation=activation)(x)
        x = Dropout(0.2)(x)

        encoder_model = Model(i, x)
        
        return encoder_model

    
