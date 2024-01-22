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
import argparse

parser = argparse.ArgumentParser(description='Argument Parser Example')

# Add the command line arguments
parser.add_argument('--is_an_initial_test', action='store_true', default=False,
                    help='Enable if it is an initial test')
parser.add_argument('--use_best_config', action='store_true', default=True,
                    help='Enable for the best configuration')
parser.add_argument('--repetitions_for_each_experiment', type=int, default=1,
                    help='Number of repetitions for each experiment')
parser.add_argument('--use_stored_model', type=int, default=True,
                    help='Wheter use or not used the weights of the stored model')


# Parse the command line arguments
args = parser.parse_args()

# Access the parsed arguments
is_an_initial_test = args.is_an_initial_test
use_best_config = args.use_best_config
repetitions_for_each_experiment = args.repetitions_for_each_experiment
use_stored_model = args.use_stored_model
#use_stored_model = False

# Print the values
print('is_an_initial_test:', is_an_initial_test)
print('use_best_config:', use_best_config)
print('repetitions_for_each_experiment:', repetitions_for_each_experiment)
print('use_stored_model:', use_stored_model)


# +
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# +
# Error with certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# +

try:
    if(__IPYTHON__):
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")
except:
    print("Not in IPython")



# + 
import os
import sys
sys.path.append(os.path.abspath("/NeurIPS-2023/src/kqm"))
os.chdir("/NeurIPS-2023/src/kqm")

from generate_grid_config_from_base_and_array_parameter import generate_grid_config_from_base_and_array_parameter
from experiment_classification_encoder_kqmd import experiment_classification_encoder_kqmd

def run_experiments(configs, architecture, experiment_function):
    with tf.device('/device:GPU:0'):
        for index_config in range(len(configs)):
            config = configs[index_config] 
            config["architecture"] = architecture
            print(config)
            experiment_function(config)

def check_whether_it_is_an_initial_test(is_an_initial_test, architecture,  configs):
    if not is_an_initial_test:
        run_experiments(configs, "kqmd_cnn", experiment_classification_encoder_kqmd)
    else: 
        only_config = configs[0]
        only_config["epochs"] = 1
        only_config["architecture"] = architecture
        with tf.device('/device:GPU:0'):
            experiment_classification_encoder_kqmd(only_config)

def generate_configs_and_check_whether_it_is_an_initial_test(is_an_initial_test, base_config, config_grid):
    configs = generate_grid_config_from_base_and_array_parameter(base_config, config_grid)
    check_whether_it_is_an_initial_test(is_an_initial_test, base_config["architecture"], configs)
    return configs



def run_experiments_using_grid_configs_with_pretrained_encoder_and_initial_prototype_modifier_multiplier(is_an_initial_test, base_config): 


    config_grid = {
        "encoded_size": [2**i for i in range(1,8)],
        "n_comp": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    }

    configs = generate_configs_and_check_whether_it_is_an_initial_test(is_an_initial_test, base_config, config_grid)

    



def run_hyperparameter_search():

    base_config_mnist = {
        "learning_rate": 5e-4,
        "optimizer": "adam",
        "architecture": "kqmd_cnn",
        "dataset": "mnist",
        "input_shape": (28, 28, 1),
        "base_depth": 32,
        "batch_size": 32,
        "num_classes": 10,
        "epochs": 50,
        "verbose": 1,
        "early_stopping_patience": 5,
        "val_size": 0.2,
        "val_random_state": 48,
        "visualization_enabled": False,
        "wandb_enabled": True,
        "additional_first_dense_encoder_layer_trained_with_cnn": False,
        "additional_first_dense_layer_size": None,
        "dense_encoder_layer_trained_with_cnn": True,
        "encoder_pretrain_enabled": True,
        "initial_prototype_modifier_multiplier": 1,
        "training": True,
        "autoencoder_pretrain_epochs": 20,
        "autoencoder_pretrain_batch_size": 64,
        "autoencoder_pretrain_verbose": 1,
        "autoencoder_shuffle": True,
        "autoencoder_early_stopping_patience": 3,


    }

    base_config_fashion_mnist = {
        "learning_rate": 5e-4,
        "optimizer": "adam",
        "architecture": "kqmd_cnn",
        "dataset": "fashion_mnist",
        "input_shape": (28, 28, 1),
        "base_depth": 32,
        "batch_size": 32,
        "num_classes": 10,
        "epochs": 50,
        "verbose": 1,
        "early_stopping_patience": 5,
        "val_size": 0.2,
        "val_random_state": 48,
        "visualization_enabled": False,
        "wandb_enabled": True, 
        "additional_first_dense_encoder_layer_trained_with_cnn": False,
        "additional_first_dense_layer_size": None,
        "dense_encoder_layer_trained_with_cnn": True,
        "encoder_pretrain_enabled": False,
        "initial_prototype_modifier_multiplier": 1,
        "training": True,
        "autoencoder_pretrain_epochs": 20,
        "autoencoder_pretrain_batch_size": 64,
        "autoencoder_pretrain_verbose": 1,
        "autoencoder_shuffle": True,
        "autoencoder_early_stopping_patience": 3,


    }

    base_config_cifar = {
        "learning_rate": 5e-4,
        "optimizer": "adam",
        "architecture": "kqmd_cnn",
        "dataset": "cifar10",
        "input_shape": (32, 32, 3),
        "base_depth": 32,
        "batch_size": 32,
        "num_classes": 10,
        "epochs": 50,
        "verbose": 1,
        "early_stopping_patience": 5,
        "val_size": 0.2,
        "val_random_state": 48,
        "visualization_enabled": False,
        "wandb_enabled": True,
        "additional_first_dense_encoder_layer_trained_with_cnn": False,
        "additional_first_dense_layer_size": None,
        "dense_encoder_layer_trained_with_cnn": True,
        "encoder_pretrain_enabled": False,
        "initial_prototype_modifier_multiplier": 1,
        "training": True,
        "autoencoder_pretrain_epochs": 20,
        "autoencoder_pretrain_batch_size": 64,
        "autoencoder_pretrain_verbose": 1,
        "autoencoder_shuffle": True,
        "autoencoder_early_stopping_patience": 3,


    }

    base_configs = [ base_config_cifar]  

    for base_config in base_configs: 
        run_experiments_using_grid_configs_with_pretrained_encoder_and_initial_prototype_modifier_multiplier(is_an_initial_test, base_config)





if use_best_config == False:
    run_hyperparameter_search()
else:
    best_config_mnist = {
            "learning_rate": 5e-4,
            "optimizer": "adam",
            "architecture": "kqmd_cnn",
            "dataset": "mnist",
            "input_shape": (28, 28, 1),
            "base_depth": 32,
            "batch_size": 32,
            "num_classes": 10,
            "epochs": 50,
            "verbose": 1,
            "early_stopping_patience": 5,
            "val_size": 0.2,
            "val_random_state": 48,
            "visualization_enabled": False,
            "wandb_enabled": True,
            "additional_first_dense_encoder_layer_trained_with_cnn": False,
            "additional_first_dense_layer_size": None,
            "dense_encoder_layer_trained_with_cnn": True,
            "encoder_pretrain_enabled": True,
            "initial_prototype_modifier_multiplier": 1,
            "training": not use_stored_model,
            "autoencoder_pretrain_epochs": 20,
            "autoencoder_pretrain_batch_size": 64,
            "autoencoder_pretrain_verbose": 1,
            "autoencoder_shuffle": True,
            "autoencoder_early_stopping_patience": 3,
            "encoded_size": 32,
            "n_comp": 1024,
            "encoder_pretrain_enabled": True,
            "initial_prototype_modifier_multiplier": 1
        }
    for i in range(repetitions_for_each_experiment):
        experiment_classification_encoder_kqmd(best_config_mnist)

    best_config_fashion_mnist = {
          "learning_rate": 5e-4,
        "optimizer": "adam",
        "architecture": "kqmd_cnn",
        "dataset": "fashion_mnist",
        "input_shape": (28, 28, 1),
        "base_depth": 32,
        "batch_size": 32,
        "num_classes": 10,
        "epochs": 50,
        "verbose": 1,
        "early_stopping_patience": 5,
        "val_size": 0.2,
        "val_random_state": 48,
        "visualization_enabled": False,
        "wandb_enabled": True, 
        "additional_first_dense_encoder_layer_trained_with_cnn": False,
        "additional_first_dense_layer_size": None,
        "dense_encoder_layer_trained_with_cnn": True,
        "encoder_pretrain_enabled": False,
        "initial_prototype_modifier_multiplier": 1,
        "training": not use_stored_model,
        "autoencoder_pretrain_epochs": 20,
        "autoencoder_pretrain_batch_size": 64,
        "autoencoder_pretrain_verbose": 1,
        "autoencoder_shuffle": True,
        "autoencoder_early_stopping_patience": 3,
        "encoded_size": 16,
        "n_comp": 1024,
        "encoder_pretrain_enabled": True,
        "initial_prototype_modifier_multiplier": 1

    }

    for i in range(repetitions_for_each_experiment):
        experiment_classification_encoder_kqmd(best_config_fashion_mnist)

    best_config_cifar10 = {
        "learning_rate": 5e-4,
        "optimizer": "adam",
        "architecture": "kqmd_cnn",
        "dataset": "cifar10",
        "input_shape": (32, 32, 3),
        "base_depth": 32,
        "batch_size": 32,
        "num_classes": 10,
        "epochs": 50,
        "verbose": 1,
        "early_stopping_patience": 5,
        "val_size": 0.2,
        "val_random_state": 48,
        "visualization_enabled": False,
        "wandb_enabled": True,
        "additional_first_dense_encoder_layer_trained_with_cnn": False,
        "additional_first_dense_layer_size": None,
        "dense_encoder_layer_trained_with_cnn": True,
        "encoder_pretrain_enabled": False,
        "initial_prototype_modifier_multiplier": 1,
        "training": not use_stored_model,
        "autoencoder_pretrain_epochs": 20,
        "autoencoder_pretrain_batch_size": 64,
        "autoencoder_pretrain_verbose": 1,
        "autoencoder_shuffle": True,
        "autoencoder_early_stopping_patience": 3,
        "encoded_size": 32,
        "n_comp": 1024,
        "encoder_pretrain_enabled": True,
        "initial_prototype_modifier_multiplier": 1

    }

    config = best_config_cifar10

    for i in range(repetitions_for_each_experiment):
        experiment_classification_encoder_kqmd(best_config_cifar10)

