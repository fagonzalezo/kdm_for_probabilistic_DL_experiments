from keras.layers import Input, Dense
from keras.models import Model
import kqm

def create_a_model_using_a_KQMUnitClassifier(input_shape, encoded_size, n_comp, num_classes, encoder, sigma, sigma_trainable=True):
    inputs = Input(shape=input_shape)
    encoded = encoder(inputs)
    kernel = kqm.RBFKernelLayer(sigma=sigma, dim=encoded_size, trainable=sigma_trainable)
    kqm_unit = kqm.KQMUnit(kernel=kernel, dim_x=encoded_size, dim_y=num_classes, n_comp=n_comp)
    rho_x = kqm.pure2dm(encoded)
    rho_y = kqm_unit(rho_x)
    probs = kqm.dm2discrete(rho_y)
    kqm_class = Model(inputs, probs)
    return kernel, kqm_unit, kqm_class


