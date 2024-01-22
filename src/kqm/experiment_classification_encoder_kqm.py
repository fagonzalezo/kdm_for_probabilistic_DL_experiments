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
from kqm_model import KQMModel
from classifier_cnn_model import classifier_cnn_model
from capture_kqm_unit_params import capture_kqm_unit_params

def experiment_classification_encoder_kqm(config):

    architecture, input_shape, base_depth, encoded_size, n_comp, sigma, num_classes, adam_lr, early_stopping_patience, initial_prototype_modifier_multiplier,\
            epochs, batch_size, verbose, wandb_enabled, encoder_pretrain_enabled, dataset, val_size, val_random_state, visualization_enabled, \
            training =\
            config["architecture"],\
            config["input_shape"], config["base_depth"], config["encoded_size"], config["n_comp"], 0.1, config["num_classes"],\
            config["learning_rate"], config["early_stopping_patience"], config["initial_prototype_modifier_multiplier"],\
            config["epochs"], config["batch_size"], config["verbose"], config["wandb_enabled"],\
            config["encoder_pretrain_enabled"], config["dataset"], config["val_size"], config["val_random_state"], config["visualization_enabled"],\
            config["training"]



    X_train, X_test, y_train, y_test, X_val, y_val, early_stopping_callback =\
            charge_dataset_and_initialize_experiment(config, architecture, dataset, input_shape, val_size, val_random_state, visualization_enabled,\
                wandb_enabled, early_stopping_patience, n_comp, encoded_size)

    if encoder_pretrain_enabled:
        encoder = encoder_model(input_shape, base_depth, encoded_size, dataset)
        cnn_model = classifier_cnn_model(input_shape, num_classes, encoder)
        compile_model(cnn_model, adam_lr)
        if(training):
            train_the_model(cnn_model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled)    
     
        checkpoint_path = "../../models/cnn-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
        if(training):
            cnn_model.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
        else:
            cnn_model.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))

        encoder_weights = encoder.get_weights()




    encoder_kqm = encoder_model(input_shape, base_depth, encoded_size, dataset)

    if encoder_pretrain_enabled:
        encoder_kqm.set_weights(encoder_weights)


    print(encoder_kqm.summary())

    kqm_model = KQMModel(encoded_size=encoded_size, encoder=encoder_kqm, n_comp=n_comp)

    compile_model(kqm_model, adam_lr)

    kqm_model.predict(X_train[:1])
    num_of_parameters_of_kqm_model = capture_kqm_unit_params(kqm_model)

    encoded_samples = initialize_prototypes_using_random_samples_from_the_training_set(kqm_model, X_train, y_train,\
            encoder_kqm, num_classes, n_comp, initial_prototype_modifier_multiplier)
    
    initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes(encoded_samples, kqm_model.kernel)

    print(kqm_model.summary())


    if(training): 
        train_the_model(kqm_model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled)    

    checkpoint_path = "../../models/kqm-classification-cnn-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    if(training):
        kqm_model.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    else:
        kqm_model.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))


    if visualization_enabled: 
        visualize_the_points_in_the_feature_space_and_plot_the_prototypes(kqm_unit, encoder_kqm, X_train, y_train)

    train_loss, train_accuracy, test_loss, test_accuracy = evaluate_the_model(kqm_model, X_test, y_test, X_train, y_train, kqm_model.kernel)



