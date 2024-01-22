import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

def train_autoencoder_model(autoencoder, X_train, X_val, autoencoder_pretrain_epochs, autoencoder_pretrain_batch_size, autoencoder_pretrain_verbose,\
            wandb_enabled, autoencoder_shuffle, autoencoder_early_stopping_patience):

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=autoencoder_early_stopping_patience, restore_best_weights=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_delta=0.0001)
    callbacks = [
      early_stopping_callback,
        reduce_lr_on_plateau
        ]


    autoencoder.fit(X_train, X_train, validation_data=(X_val[:1000], X_val[:1000]), epochs=autoencoder_pretrain_epochs,\
            batch_size=autoencoder_pretrain_batch_size, verbose=autoencoder_pretrain_verbose, shuffle=autoencoder_shuffle,
                    callbacks=callbacks, )


