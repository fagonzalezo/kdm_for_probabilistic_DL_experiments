import matplotlib.pyplot as plt
import numpy as np

def print_random_images(X_train, y_train):
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        idx = np.random.randint(0, X_train.shape[0])
        plt.grid(False)
        plt.imshow(X_train[idx], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[idx]])


