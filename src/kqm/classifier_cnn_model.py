from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def classifier_cnn_model(input_shape, num_classes, encoder):
    i = Input(shape=input_shape)

    x = encoder(i)

    # Output layer
    x = Dense(num_classes)(x)

    model = Model(i, x)
    return model
