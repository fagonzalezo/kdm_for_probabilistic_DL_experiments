import tensorflow as tf
tfkl = tf.keras.layers

def attach_softmax_layer_to_encoder(encoder, num_classes):
    return tf.keras.Sequential([
            encoder,
            tfkl.Flatten(),
            tfkl.Dense(num_classes) 
        ])

