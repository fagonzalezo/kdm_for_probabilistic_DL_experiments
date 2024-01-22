import tensorflow as tf
tfkl = tf.keras.layers


def encoded_decoded():
    encoder = tf.keras.Sequential([
            tfkl.Conv2D(64, (3, 3), padding='same'),
            tfkl.BatchNormalization(),
            tfkl.Activation('relu'),
            tfkl.MaxPooling2D((2, 2), padding='same'),
            tfkl.Conv2D(32, (3, 3), padding='same'),
            tfkl.BatchNormalization(),
            tfkl.Activation('relu'),
            tfkl.MaxPooling2D((2, 2), padding='same'),
            tfkl.Conv2D(16, (3, 3), padding='same'),
            tfkl.BatchNormalization(),
            tfkl.Activation('relu'),
            tfkl.MaxPooling2D((2, 2), padding='same')])

    decoder = tf.keras.Sequential([
        tfkl.Conv2D(16, (3, 3), padding='same'),
        tfkl.BatchNormalization(),
        tfkl.Activation('relu'),
        tfkl.UpSampling2D((2, 2)),
        tfkl.Conv2D(32, (3, 3), padding='same'),
        tfkl.BatchNormalization(),
        tfkl.Activation('relu'),
        tfkl.UpSampling2D((2, 2)),
        tfkl.Conv2D(64, (3, 3), padding='same'),
        tfkl.BatchNormalization(),
        tfkl.Activation('relu'),
        tfkl.UpSampling2D((2, 2)),
        tfkl.Conv2D(3, (3, 3), padding='same'),
        tfkl.BatchNormalization(),
        tfkl.Activation('sigmoid')])

    return encoder, decoder, 256


