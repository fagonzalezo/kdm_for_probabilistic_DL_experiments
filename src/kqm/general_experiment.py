# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: tf2_py39
#     language: python
#     name: python3
# ---

# +
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# +
# Error with certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# +
%load_ext autoreload
%autoreload 2

# + 
import os
import sys
sys.path.append(os.path.abspath("/NeurIPS-2023/src/kqm"))
os.chdir("/NeurIPS-2023/src/kqm")


base_config_mnist = {
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "architecture": "kqm_cnn",
    "dataset": "mnist",
    "input_shape": (28, 28, 1),
    "base_depth": 32,
    "batch_size": 32,
    "num_classes": 10,
    "epochs": 10,
    "verbose": 1,
    "early_stopping_patience": 5,
    "val_size": 0.2,
    "val_random_state": 48,
    "visualization_enabled": False,
    "wandb_enabled": False,
    "additional_first_dense_encoder_layer_trained_with_cnn": False,
    "additional_first_dense_layer_size": None,
    "dense_encoder_layer_trained_with_cnn": True,
    "encoder_pretrain_enabled": False,
    "initial_prototype_modifier_multiplier": 1

}

base_config_fashion_mnist = {
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "architecture": "kqm_cnn",
    "dataset": "fashion_mnist",
    "input_shape": (28, 28, 1),
    "base_depth": 32,
    "batch_size": 32,
    "num_classes": 10,
    "epochs": 10,
    "verbose": 1,
    "early_stopping_patience": 5,
    "val_size": 0.2,
    "val_random_state": 48,
    "visualization_enabled": False,
    "wandb_enabled": False, 
    "additional_first_dense_encoder_layer_trained_with_cnn": False,
    "additional_first_dense_layer_size": None,
    "dense_encoder_layer_trained_with_cnn": True,
    "encoder_pretrain_enabled": False,
    "initial_prototype_modifier_multiplier": 100
}




base_config_cifar = {
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "architecture": "kqm_cnn",
    "dataset": "cifar10",
    "input_shape": (32, 32, 3),
    "base_depth": 32,
    "batch_size": 32,
    "num_classes": 10,
    "epochs": 10,
    "verbose": 1,
    "early_stopping_patience": 5,
    "val_size": 0.2,
    "val_random_state": 48,
    "visualization_enabled": False,
    "wandb_enabled": False,
    "additional_first_dense_encoder_layer_trained_with_cnn": False,
    "additional_first_dense_layer_size": None,
    "dense_encoder_layer_trained_with_cnn": True,
    "initial_prototype_modifier_multiplier": 1
}


#base_config = base_config_mnist
base_config = base_config_fashion_mnist
#base_config = base_config_cifar



#config_grid = {
#    "encoded_size": [2**i for i in range(1,8)],
#    "n_comp": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
#}

config_grid = {
    "encoded_size": [128],
    "n_comp": [128],
}

from generate_grid_config_from_base_and_array_parameter import generate_grid_config_from_base_and_array_parameter
from experiment_classification_encoder_kqm import experiment_classification_encoder_kqm

configs = generate_grid_config_from_base_and_array_parameter(base_config, config_grid)

config = configs[-1]

experiment_classification_encoder_kqm(configs[0])


print(tf.config.list_physical_devices('GPU'))
with tf.device('/device:GPU:0'):

    for index_config in range(len(configs)):
        print(configs[index_config])
        experiment_classification_encoder_kqm(configs[index_config])





