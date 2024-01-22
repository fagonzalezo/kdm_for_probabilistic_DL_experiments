import numpy as np
def preprocess_data(X_train, X_test, y_train, y_test, num_classes):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train,axis=(0,1,2,3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    #y_train = tf.keras.utils.to_categorical(y_train,num_classes)
    #y_test = tf.keras.utils.to_categorical(y_test,num_classes)
    return X_train, X_test, y_train, y_test


