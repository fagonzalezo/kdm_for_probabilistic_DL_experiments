import numpy as np
import tensorflow as tf
def initialize_prototypes_using_random_samples_from_the_training_set(kqm_model, X_train, y_train, encoder, num_classes, n_comp, initial_prototype_modifier_multiplier): 
    kqm_model.predict(X_train[:1])
    idx = np.random.randint(X_train.shape[0], size=n_comp)
    encoded_samples = encoder(X_train[idx]) * initial_prototype_modifier_multiplier
    kqm_model.kqm_unit.c_x.assign(encoded_samples)
    kqm_model.kqm_unit.c_y.assign(tf.one_hot(y_train[idx], num_classes))
    return encoded_samples


