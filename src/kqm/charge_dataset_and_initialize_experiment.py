import tensorflow as tf 
from charge_dataset import charge_dataset
from preprocess_data import preprocess_data
from partitioning_in_training_and_testing import partitioning_in_training_and_testing
from initializing_wand_experiment_to_track_experiment import initializing_wand_experiment_to_track_experiment
from print_random_images import print_random_images

def charge_dataset_and_initialize_experiment(config, architecture, dataset, input_shape, val_size, val_random_state, visualization_enabled,\
        wandb_enabled, early_stopping_patience, n_comp, encoded_size):
    X_train, X_test, y_train, y_test = charge_dataset(dataset, input_shape)
    #X_train, X_val, y_train, y_val = preprocess_data(X_train, X_test, y_train, y_test, config["num_classes"])
    X_train, X_val, y_train, y_val = partitioning_in_training_and_testing(X_train, y_train, val_size, val_random_state)

    if visualization_enabled: 
        print_random_images(X_train, y_train)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping_patience)


    return X_train, X_test, y_train, y_test, X_val, y_val, early_stopping_callback
