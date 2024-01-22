from keras.models import Model
import tensorflow as tf
tfkl = tf.keras.layers

class Autoencoder(Model):
  def __init__(self, encoder, decoder, decoder_input_size, dense_layer_features=None, encoded_size=None):
    super(Autoencoder, self).__init__()
    self.encoder = encoder
    self.decoder_input_size = decoder_input_size
    if dense_layer_features:
      self.dense_layer = tfkl.Dense(dense_layer_features)
    if encoded_size:
      self.flatten = tfkl.Flatten()
      self.final_encoded_layer = tfkl.Dense(encoded_size)
      self.final_decoded_layer = tfkl.Dense(encoded_size)
    self.decoder = decoder

  def call(self, x):
    encoded = self.encoder(x)
    if hasattr(self, 'dense_layer'):
      encoded = self.dense_layer(encoded)
    if hasattr(self, 'final_encoded_layer'):
      encoded = self.flatten(encoded)
      encoded = self.final_encoded_layer(encoded)
    decoded = self.decoder(encoded)
    return decoded

  def get_full_encoder(self):
    list_encoder = [self.encoder]
    if hasattr(self, 'dense_layer'):
        list_encoder.append(self.dense_layer)
    if hasattr(self, 'final_encoded_layer'):
        list_encoder.append(self.final_encoded_layer)
 
    return tf.keras.Sequential(list_encoder)

