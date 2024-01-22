
from keras.losses import mean_squared_error
import keras.backend as K

def loss_function(y_true, y_pred):  ## loss function for using in autoencoder models
    mses = mean_squared_error(y_true, y_pred)
    return K.sum(mses, axis=(1,2))

from capture_config import capture_config
from charge_dataset_and_initialize_experiment import charge_dataset_and_initialize_experiment
from create_a_deep_encoder import create_a_deep_encoder
from compile_model import compile_model
from initialize_prototypes_using_random_samples_from_the_training_set import initialize_prototypes_using_random_samples_from_the_training_set
from initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes import initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes
from train_the_model import train_the_model
from visualize_the_points_in_the_feature_space_and_plot_the_prototypes import visualize_the_points_in_the_feature_space_and_plot_the_prototypes
from evaluate_the_model import evaluate_the_model
from capture_sigma_parameter_and_update_it_on_wandb import capture_sigma_parameter_and_update_it_on_wandb
from finish_experiment import finish_experiment
from encoder_model import encoder_model
from decoder_model import decoder_model
from kqm_model import KQMModel
from classifier_cnn_model import classifier_cnn_model
from capture_kqm_unit_params import capture_kqm_unit_params
import wandb
from create_autoencoder import create_autoencoder,  end_to_end
from compile_autoencoder_model import compile_autoencoder_model
from train_autoencoder_model import train_autoencoder_model
from show_original_and_reconstructed_image import show_original_and_reconstructed_image
from kqm import KQMDenEstModel2
from kqm_gen_model import KQMGenModel
from kqm import dm2distrib


from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

def experiment_generative_kqm(config):

    print(config)
    architecture, input_shape, base_depth, encoded_size, n_comp, sigma, num_classes, learning_rate, early_stopping_patience, initial_prototype_modifier_multiplier,\
            epochs, batch_size, verbose, wandb_enabled, encoder_pretrain_enabled, dataset, val_size, val_random_state, visualization_enabled,\
            autoencoder_pretrain_epochs, autoencoder_pretrain_batch_size, autoencoder_pretrain_verbose, autoencoder_shuffle, autoencoder_early_stopping_patience =\
            config["architecture"],\
            config["input_shape"], config["base_depth"], config["encoded_size"], config["n_comp"], 0.1, config["num_classes"],\
            config["learning_rate"], config["early_stopping_patience"], config["initial_prototype_modifier_multiplier"],\
            config["epochs"], config["batch_size"], config["verbose"], config["wandb_enabled"],\
            config["encoder_pretrain_enabled"], config["dataset"], config["val_size"], config["val_random_state"], config["visualization_enabled"],\
            config["autoencoder_pretrain_epochs"], config["autoencoder_pretrain_batch_size"], config["autoencoder_pretrain_verbose"], config["autoencoder_shuffle"],\
            config["autoencoder_early_stopping_patience"]



    X_train, X_test, y_train, y_test, X_val, y_val, early_stopping_callback =\
            charge_dataset_and_initialize_experiment(config, architecture, dataset, input_shape, val_size, val_random_state, visualization_enabled,\
                wandb_enabled, early_stopping_patience, n_comp, encoded_size)


    y_train_autoencoder = tf.keras.utils.to_categorical(y_train, 10)
    y_val_autoencoder = tf.keras.utils.to_categorical(y_val, 10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)
    multimodel, encoder, decoder = end_to_end()
    #multimodel= end_to_end()
    multimodel.compile(loss = {'classification': 'categorical_crossentropy', 'autoencoder': loss_function}, 
                      loss_weights = {'classification': 0.9, 'autoencoder': 0.1}, 
                      optimizer = SGD(lr= 0.01, momentum= 0.9),
                      metrics = {'classification': ['accuracy'], 'autoencoder': []})
    


    
    callbacks = [er, lr]
    hist_mul = multimodel.fit(X_train, [X_train,y_train_autoencoder], batch_size=512, epochs=100, 
                              validation_data = (X_val, [X_val, y_val_autoencoder]),
                              shuffle=True, callbacks=callbacks)
    #                           class_weight=class_weights

    checkpoint_path = "../../models/cnn-autoencoder-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    multimodel.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    multimodel.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))

    decoded_images_test = multimodel.predict(X_test[:10])
    show_original_and_reconstructed_image(X_test[:10], decoded_images_test[0], input_shape)


