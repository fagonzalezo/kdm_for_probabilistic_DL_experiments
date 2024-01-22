import tensorflow as tf
tfkl = tf.keras.layers

def create_a_deep_encoder_with_max_polling(input_shape, base_depth, encoded_size):
    encoder = tf.keras.Sequential([
        tfkl.InputLayer(input_shape=input_shape),
        tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        tfkl.Conv2D(base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.MaxPooling2D((2, 2)),

        tfkl.Conv2D(2 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(2 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(2 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.MaxPooling2D((2, 2)),


        tfkl.Conv2D(4 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(4 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.Conv2D(4 * base_depth, 5, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
        tfkl.MaxPooling2D((2, 2)),


        tfkl.Flatten(),
        tfkl.Dense(encoded_size),#, activity_regularizer=tf.keras.regularizers.l2(1e-3)),
        #tfk.layers.LayerNormalization(),
    ])
    return encoder


