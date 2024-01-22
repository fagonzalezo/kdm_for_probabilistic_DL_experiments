import tensorflow as tf
tfkl = tf.keras.layers

def create_a_deep_encoder(input_shape, base_depth, encoded_size):
    encoder = tf.keras.Sequential([
        tfkl.InputLayer(input_shape=input_shape),
        tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        tfkl.Conv2D(base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(base_depth, 5, strides=2,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(2 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(2 * base_depth, 5, strides=2,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(4 * encoded_size, 7, strides=1,
                    padding='valid', activation=tf.nn.leaky_relu),
        tfkl.Flatten(),
        #tfkl.Dense(1000),#, activity_regularizer=tf.keras.regularizers.l2(1e-3)),
        tfkl.Dense(encoded_size)   #tfk.layers.LayerNormalization(),
    ])
    return encoder


