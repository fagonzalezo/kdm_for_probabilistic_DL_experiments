import tensorflow as tf
def charge_dataset(dataset, input_shape):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    # reshape the data to include a channel dimension
    X_train = X_train.reshape([X_train.shape[0]] + list(input_shape))
    X_test = X_test.reshape([X_test.shape[0]] + list(input_shape))

    if dataset == "cifar10":
        y_train = tf.squeeze(y_train, axis=1).numpy()
        y_test = tf.squeeze(y_test, axis=1).numpy()

    return X_train, X_test, y_train, y_test


