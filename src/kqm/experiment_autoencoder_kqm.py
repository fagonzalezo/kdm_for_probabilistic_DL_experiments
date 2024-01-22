from capture_config import capture_config
from charge_dataset_and_initialize_experiment import charge_dataset_and_initialize_experiment
from autoencoder import  Autoencoder
from encoded_decoded import encoded_decoded
from compile_autoencoder_model import compile_autoencoder_model
from train_autoencoder_model import train_autoencoder_model
from show_original_and_reconstructed_image import show_original_and_reconstructed_image
from encoded_decoded import encoded_decoded
from attach_dense_layer_to_encoder import attach_dense_layer_to_encoder
from create_a_model_using_a_KQMUnitClassifier import create_a_model_using_a_KQMUnitClassifier
from compile_model import compile_model
from initialize_prototypes_using_random_samples_from_the_training_set import initialize_prototypes_using_random_samples_from_the_training_set
from initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes import initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes
from train_the_model import train_the_model
from capture_sigma_parameter_and_update_it_on_wandb import capture_sigma_parameter_and_update_it_on_wandb
from visualize_the_points_in_the_feature_space_and_plot_the_prototypes import visualize_the_points_in_the_feature_space_and_plot_the_prototypes
from finish_experiment import finish_experiment
from evaluate_the_model import evaluate_the_model

def experiment_autoencoder_model(config):
     
    input_shape, base_depth, encoded_size, n_comp, sigma, num_classes, adam_lr,\
            epochs, batch_size, verbose, wandb_enabled = capture_config(config)

    X_train, X_test, y_train, y_test, X_val, y_val, early_stopping_callback =\
            charge_dataset_and_initialize_experiment(config)

    encoder, decoder, decoder_input_size = encoded_decoded()
     
    if config["additional_first_dense_encoder_layer_trained_with_cnn"]:
        additional_first_dense_layer_size = config["additional_first_dense_layer_size"]
    else:
        additional_first_dense_layer_size = None

    if config["dense_encoder_layer_trained_with_cnn"]:
        autoencoder_model = Autoencoder(encoder, decoder, decoder_input_size, dense_layer_features = additional_first_dense_layer_size, encoded_size=encoded_size) 
    else:
        autoencoder_model = Autoencoder(encoder, decoder, decoder_input_size, dense_layer_features = additional_first_dense_layer_size, encoded_size=None) 


    compile_autoencoder_model(autoencoder_model, adam_lr)

    train_autoencoder_model(autoencoder_model, X_train, X_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled)    

    print(autoencoder_model.summary())

    decoded_images_test = autoencoder_model.predict(X_test[:10])

    show_original_and_reconstructed_image(X_test[:10], decoded_images_test, input_shape)

    if config["dense_encoder_layer_trained_with_cnn"]:
        full_encoder = attach_dense_layer_to_encoder(autoencoder_model.encoder, encoded_size=None)
    else:
        full_encoder = attach_dense_layer_to_encoder(autoencoder_model.encoder, encoded_size=encoded_size)


    kernel, kqm_unit, kqm_class = create_a_model_using_a_KQMUnitClassifier(input_shape, encoded_size, n_comp, num_classes, full_encoder, sigma, True) 

    print(kqm_class.summary())

    encoder.trainable = False
    #encoder.trainable = True

    compile_model(kqm_class, adam_lr)
    
    idx = initialize_prototypes_using_random_samples_from_the_training_set(kqm_unit, X_train, y_train, full_encoder, num_classes, n_comp)
    
    initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes(full_encoder, X_train, idx, kernel)

    train_the_model(kqm_class, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled)    

    kqm_class.predict(X_train[:10])

    if wandb_enabled: 
        capture_sigma_parameter_and_update_it_on_wandb(sigma, config, kernel)

    if config["visualization_enabled"]: 
        visualize_the_points_in_the_feature_space_and_plot_the_prototypes(kqm_unit, full_encoder, X_train, y_train)

    train_loss, train_accuracy, test_loss, test_accuracy = evaluate_the_model(kqm_class, X_test, y_test, X_train, y_train, kernel)

    if wandb_enabled: 
        finish_experiment(train_loss, train_accuracy, test_loss, test_accuracy)


