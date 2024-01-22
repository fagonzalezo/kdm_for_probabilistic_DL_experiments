import tensorflow as tf
tfkl = tf.keras.layers

def attach_dense_layer_to_encoder(encoder, encoded_size=None):
    layers =[
            encoder,
            tfkl.Flatten()
        ]
    if encoded_size:
        layers.append(tfkl.Dense(encoded_size)) 
    return tf.keras.Sequential(layers)

