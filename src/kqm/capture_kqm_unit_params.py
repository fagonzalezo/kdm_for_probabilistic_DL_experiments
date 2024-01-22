# Purpose: Capture the number of params in the KQM unit from a KQM model.

import numpy as np
def capture_kqm_unit_params(kqm_model):
    kqm_shape = kqm_model.kqm_unit.trainable_weights
    return np.sum([np.prod(v.get_shape()) for v in kqm_shape]) + 1 # rbflayer
