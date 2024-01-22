from keras import optimizers
from tensorflow.keras.losses import MeanSquaredError

def compile_autoencoder_model(model, adam_lr):
    model.compile(optimizer=optimizers.Adam(learning_rate=adam_lr),
                 loss=MeanSquaredError())


