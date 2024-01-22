from keras.callbacks import ReduceLROnPlateau

def train_the_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, verbose, early_stopping_callback, wandb_enabled):

    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_delta=0.0001)

    callbacks = [
        early_stopping_callback,
        reduce_lr_on_plateau
    ]
 

    model.fit(X_train, y_train, validation_data=(X_val[:1000], y_val[:1000]), epochs=epochs,\
            batch_size=batch_size, verbose=verbose, callbacks=callbacks)


