import numpy as np
import matplotlib.pyplot as plt
def visualize_the_points_in_the_feature_space_and_plot_the_prototypes(kqm_unit, encoder, X_train, y_train):
    idx = np.random.randint(X_train.shape[0], size=1000)
    plt.scatter(encoder(X_train[idx,:])[:, 0], encoder(X_train[idx,:])[:, 1], alpha=0.5, c=y_train[idx], cmap=plt.cm.coolwarm) # points
    plt.scatter(kqm_unit.c_x.numpy()[:, 0], kqm_unit.c_x.numpy()[:, 1], c='k', marker='X', s=50) # prototypes
