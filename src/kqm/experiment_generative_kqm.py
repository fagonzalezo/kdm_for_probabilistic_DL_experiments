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
from create_autoencoder import create_autoencoder
from compile_autoencoder_model import compile_autoencoder_model
from train_autoencoder_model import train_autoencoder_model
from show_original_and_reconstructed_image import show_original_and_reconstructed_image
from kqm import KQMDenEstModel2
from kqm_gen_model import KQMGenModel
from kqm import dm2distrib
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


def experiment_generative_kqm(config):

    print(config)
    architecture, input_shape, base_depth, encoded_size, n_comp, sigma, num_classes, learning_rate, early_stopping_patience, initial_prototype_modifier_multiplier,\
            epochs, batch_size, verbose, wandb_enabled, encoder_pretrain_enabled, dataset, val_size, val_random_state, visualization_enabled, training,\
            autoencoder_pretrain_epochs, autoencoder_pretrain_batch_size, autoencoder_pretrain_verbose, autoencoder_shuffle, autoencoder_early_stopping_patience =\
            config["architecture"],\
            config["input_shape"], config["base_depth"], config["encoded_size"], config["n_comp"], 0.1, config["num_classes"],\
            config["learning_rate"], config["early_stopping_patience"], config["initial_prototype_modifier_multiplier"],\
            config["epochs"], config["batch_size"], config["verbose"], config["wandb_enabled"],\
            config["encoder_pretrain_enabled"], config["dataset"], config["val_size"], config["val_random_state"], config["visualization_enabled"],config["training"], \
            config["autoencoder_pretrain_epochs"], config["autoencoder_pretrain_batch_size"], config["autoencoder_pretrain_verbose"], config["autoencoder_shuffle"],\
            config["autoencoder_early_stopping_patience"]



    X_train, X_test, y_train, y_test, X_val, y_val, early_stopping_callback =\
            charge_dataset_and_initialize_experiment(config, architecture, dataset, input_shape, val_size, val_random_state, visualization_enabled,\
                wandb_enabled, early_stopping_patience, n_comp, encoded_size)

    encoder = encoder_model(input_shape, base_depth, encoded_size, dataset, encoder_type="generative")
    decoder = decoder_model(input_shape, base_depth, encoded_size, dataset, encoder_type="generative")

    autoencoder = create_autoencoder(encoder, decoder)

    compile_autoencoder_model(autoencoder, learning_rate)

    if(training):
        train_autoencoder_model(autoencoder, X_train, X_val, autoencoder_pretrain_epochs, autoencoder_pretrain_batch_size, autoencoder_pretrain_verbose,\
                wandb_enabled, autoencoder_shuffle, autoencoder_early_stopping_patience)



    checkpoint_path = "../../models/autoencoder-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    if(training):
        autoencoder.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    else:
        autoencoder.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))

    decoded_images_test = autoencoder.predict(X_test[:10])
    show_original_and_reconstructed_image(X_test[:10], decoded_images_test, input_shape)

    encoder_weights = encoder.get_weights()
    encoder_kqm = encoder_model(input_shape, base_depth, encoded_size, dataset, encoder_type="generative")
    encoder_kqm.set_weights(encoder_weights)
    encoder_kqm.trainable = False

    for layer in encoder_kqm.layers:
        layer.trainable = False

    print(encoder_kqm.summary())

    kqm_model = KQMModel(encoded_size=encoded_size, encoder=encoder_kqm, n_comp=n_comp)

    compile_model(kqm_model, learning_rate)

    kqm_model.predict(X_train[:1])
    num_of_parameters_of_kqm_model = capture_kqm_unit_params(kqm_model)

    encoded_samples = initialize_prototypes_using_random_samples_from_the_training_set(kqm_model, X_train, y_train,\
            encoder_kqm, num_classes, n_comp, initial_prototype_modifier_multiplier)
    
    initialize_kernel_sigma_parameter_using_pairwise_distance_of_prototypes(encoded_samples, kqm_model.kernel)

    print(kqm_model.summary())

    if(training):
        train_the_model(kqm_model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled)    

    encoder_kqm.trainable = True
    for layer in encoder_kqm.layers:
        layer.trainable = True
    compile_model(kqm_model, learning_rate*0.1)
    if(training):
        train_the_model(kqm_model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled)    


    checkpoint_path = "../../models/kqm-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    if(training):
        kqm_model.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    else:
        kqm_model.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))

    if visualization_enabled: 
        visualize_the_points_in_the_feature_space_and_plot_the_prototypes(kqm_unit, encoder_kqm, X_train, y_train)

    train_loss, train_accuracy, test_loss, test_accuracy = evaluate_the_model(kqm_model, X_test, y_test, X_train, y_train, kqm_model.kernel)



    # join X and y using a one-hot encoding for y
    enc_x = encoder_kqm.predict(X_train)
    Xy_train = np.concatenate((enc_x, np.eye(10)[y_train]), axis=1)

    kqmd_model2 = KQMDenEstModel2(encoded_size, 10, 
                                  kqm_model.kernel.sigma.numpy(), 
                                  n_comp=n_comp,
                                  trainable_sigma=True,
                                  min_sigma=kqm_model.kernel.sigma.numpy() * 0.5)

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    kqmd_model2.compile(optimizer=optimizer)
    kqmd_model2.predict(Xy_train[0:1]) # initialize the model

    # Assign the prototypes
    idx = np.random.randint(Xy_train.shape[0], size=n_comp)
    kqmd_model2.kqmover.c_x.assign(Xy_train[idx])

    print(f'Initial sigma: {kqmd_model2.kernel_x.sigma.numpy()}')

    #kqmd_model2.kqmover.c_x.assign(tf.concat([kqm_unit.c_x, kqm_unit.c_y], axis=1))
    if(training):
        kqmd_model2.fit(Xy_train, epochs=10, verbose=1, batch_size=32)

    checkpoint_path = "../../models/kqmd-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    if(training):
        kqmd_model2.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    else:
        kqmd_model2.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))



    kqm_model2 = KQMModel(encoded_size=encoded_size,
                          encoder=encoder_kqm, 
                          n_comp=n_comp)

    kqm_model2.predict(X_train[:1])
    kqm_model2.kqm_unit.c_x.assign(kqmd_model2.kqmover.c_x[:, :encoded_size])
    kqm_model2.kqm_unit.c_y.assign(kqmd_model2.kqmover.c_x[:, encoded_size:])
    kqm_model2.kqm_unit.comp_w.assign(kqmd_model2.kqmover.comp_w)
    kqm_model2.kernel.sigma.assign(kqmd_model2.kernel_x.sigma.numpy())

    kqm_model2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    kqm_model2.evaluate(X_test, y_test)

    plt.plot(np.abs(kqm_model2.kqm_unit.comp_w.numpy()))

    print(kqm_model2.kqm_unit.c_y.numpy)


    # Create the autoencoder model
    encoder_kqm.trainable = False
    autoencoder_kqm = create_autoencoder(encoder_kqm, decoder)

    # Compile the autoencoder model
    compile_autoencoder_model(autoencoder_kqm, adam_lr=5e-3)
     
    if(training):
        history = autoencoder_kqm.fit(
            X_train, X_train,
            epochs=20,
            batch_size=64,
            shuffle=True,
            validation_data=(X_test, X_test)
        )


    checkpoint_path = "../../models/autoencoder-kqm-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    if(training):
        autoencoder_kqm.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    else:
        autoencoder_kqm.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))


    decoded_images_test = autoencoder.predict(X_test[:10])
    show_original_and_reconstructed_image(X_test[:10], decoded_images_test, input_shape)

    kqmgen_model = KQMGenModel(encoded_size, n_comp=n_comp)



    kqmgen_model.predict(tf.eye(10))
    kqmgen_model.kqm_unit.c_x.assign(kqm_model2.kqm_unit.c_y)
    kqmgen_model.kqm_unit.c_y.assign(kqm_model2.kqm_unit.c_x)
    kqmgen_model.kqm_unit.comp_w.assign(kqm_model2.kqm_unit.comp_w);


    checkpoint_path = "../../models/kqmgen-{architecture}-{dataset}-{n_comp}-{encoded_size}.ckpt"
    if(training):
        kqmgen_model.save_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))
    else:
        kqmgen_model.load_weights(checkpoint_path.format(architecture=architecture, dataset=dataset, n_comp=n_comp, encoded_size=encoded_size))


    new_sigma = kqm_model2.kernel.sigma / 2
    # Generate a set of distributions from the model for each digit
    distribs = dm2distrib(kqmgen_model(tf.eye(10)), sigma=new_sigma)

    # Sample from the distributions
    for ii in range(10):
        sample = distribs[ii].sample(10).numpy()
        # Show the decoded images
        decoded_imgs = decoder(sample)
        
        fig, ax = plt.subplots(2, 5, figsize=(5,2))
        k = 0
        for i in range(2):
            for j in range(5):
                ax[i][j].imshow(decoded_imgs[k], aspect='auto', cmap='gray',  interpolation='nearest')
                k += 1

        plt.savefig(f"../../figures/{architecture}-{dataset}-{n_comp}-{encoded_size}-{ii}-{new_sigma:.3f}.eps")
        plt.savefig(f"../../figures/{architecture}-{dataset}-{n_comp}-{encoded_size}-{ii}-{new_sigma:.3f}.png")
        #plt.show()

    print(f'Final sigma: {kqmd_model2.kernel_x.sigma.numpy()}')


