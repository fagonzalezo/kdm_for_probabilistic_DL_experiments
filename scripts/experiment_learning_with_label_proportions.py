# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: tf2_py39
#     language: python
#     name: python3
# ---

# +
import argparse

parser = argparse.ArgumentParser(description='Argument Parser Example')

# Add the command line arguments
parser.add_argument('--is_an_initial_test', action='store_true', default=False,
                    help='Enable if it is an initial test')
parser.add_argument('--use_best_config', action='store_true', default=True,
                    help='Enable for the best configuration')
parser.add_argument('--repetitions_for_each_experiment', type=int, default=1,
                    help='Number of repetitions for each experiment')
parser.add_argument('--use_stored_model', type=int, default=True,
                    help='Wheter use or not used the weights of the stored model')


# Parse the command line arguments
args = parser.parse_args()

# Access the parsed arguments
is_an_initial_test = args.is_an_initial_test
use_best_config = args.use_best_config
repetitions_for_each_experiment = args.repetitions_for_each_experiment
use_stored_model = args.use_stored_model

# Print the values
print('is_an_initial_test:', is_an_initial_test)
print('use_best_config:', use_best_config)
print('repetitions_for_each_experiment:', repetitions_for_each_experiment)
print('use_stored_model:', use_stored_model)

# +
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# +
# Error with certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# +

try:
    if(__ipython__):
        from ipython import get_ipython
        ipython = get_ipython()
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")
except:
    print("not in ipython")

# + 

import os
import sys
sys.path.append(os.path.abspath("/NeurIPS-2023/src/kqm"))
os.chdir("/NeurIPS-2023/src/kqm")

 

import tensorflow as tf
import tensorflow_probability as tfp
#import tensorflow_addons as tfa
import numpy as np
tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions

"""# KQM

## Utils
"""

@tf.function
def dm2comp(dm):
    '''
    Extract vectors and weights from a factorized density matrix representation
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    '''
    return dm[:, :, 0], dm[:, :, 1:]

@tf.function
def comp2dm(w, v):
    '''
    Construct a factorized density matrix from vectors and weights
    Arguments:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    Returns:
     dm: tensor of shape (bs, n, d + 1)
    '''
    return tf.concat((w[:, :, tf.newaxis], v), axis=2)

@tf.function
def pure2dm(psi):
    '''
    Construct a factorized density matrix to represent a pure state
    Arguments:
     psi: tensor of shape (bs, d)
    Returns:
     dm: tensor of shape (bs, 1, d + 1)
    '''
    ones = tf.ones_like(psi[:, 0:1])
    dm = tf.concat((ones[:,tf.newaxis, :],
                    psi[:,tf.newaxis, :]),
                   axis=2)
    return dm

@tf.function
def dm2distrib(dm, sigma):
    '''
    Creates a Gaussian mixture distribution from the components of a density
    matrix with an RBF kernel 
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
     sigma: sigma parameter of the RBF kernel 
    Returns:
     gm: mixture of Gaussian distribution with shape (bs, )
    '''
    w, v = dm2comp(dm)
    gm = tfd.MixtureSameFamily(reparameterize=True,
            mixture_distribution=tfd.Categorical(
                                    probs=w),
            components_distribution=tfd.Independent( tfd.Normal(
                    loc=v,  # component 2
                    scale=sigma * np.sqrt(2.)),
                    reinterpreted_batch_ndims=1))
    return gm

@tf.function 
def pure_dm_overlap(x, dm, kernel):
    '''
    Calculates the overlap of a state  \phi(x) with a density 
    matrix in a RKHS defined by a kernel
    Arguments:
      x: tensor of shape (bs, d)
     dm: tensor of shape (bs, n, d + 1)
     kernel: kernel function 
              k: (bs, d) x (bs, n, d) -> (bs, n)
    Returns:
     overalp: tensor wit shape (bs, )
    '''
    w, v = dm2comp(dm)
    overlap = tf.einsum('...i,...i->...', w, kernel(x, v) ** 2)
    return overlap

"""## Kernels"""

def create_rbf_kernel(sigma):
    '''
    Builds a function that calculates the rbf kernel between two set of vectors
    Arguments:
        sigma: RBF scale parameter
    Returns:
        a function that receives 2 tensors with the following shapes
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
    '''
    @tf.function
    def reshape(A, B):
        A = tf.expand_dims(A, axis=2)
        B = tf.expand_dims(tf.expand_dims(B, axis=0), axis=0)
        return A, B
    @tf.function
    def rbf_kernel(A, B):
        A, B = reshape(A, B)
        diff = A - B
        K = tf.exp(-tf.norm(diff, axis=-1) ** 2 / (2 * sigma ** 2))
        return K
    return rbf_kernel

"""## Layers and models"""

def l1_loss(vals):
    '''
    Calculate the l1 loss for a batch of vectors
    Arguments:
        vals: tensor with shape (b_size, n)
    '''
    b_size = tf.cast(tf.shape(vals)[0], dtype=tf.float32)
    vals = vals / tf.norm(vals, axis=1)[:, tf.newaxis]
    loss = tf.reduce_sum(tf.abs(vals)) / b_size
    return loss

class KQMUnit(tf.keras.layers.Layer):
    """Kernel Quantum Measurement Unit
    Receives as input a factored density matrix represented by a set of vectors
    and weight values. 
    Returns a resulting factored density matrix.
    Input shape:
        (batch_size, n_comp_in, dim_x + 1)
        where dim_x is the dimension of the input state
        and n_comp_in is the number of components of the input factorization. 
        The weights of the input factorization of sample i are [i, :, 0], 
        and the vectors are [i, :, 1:dim_x + 1].
    Output shape:
        (batch_size, n_comp, dim_y)
        where dim_y is the dimension of the output state
        and n_comp is the number of components used to represent the train
        density matrix. The weights of the
        output factorization for sample i are [i, :, 0], and the vectors
        are [i, :, 1:dim_y + 1].
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        x_train: bool. Whether to train or not the x compoments of the train
                       density matrix.
        x_train: bool. Whether to train or not the y compoments of the train
                       density matrix.
        w_train: bool. Whether to train or not the weights of the compoments 
                       of the train density matrix. 
        n_comp: int. Number of components used to represent 
                 the train density matrix
        l1_act: float. Coefficient of the regularization term penalizing the l1
                       norm of the activations.
        l1_x: float. Coefficient of the regularization term penalizing the l1
                       norm of the x components.
        l1_y: float. Coefficient of the regularization term penalizing the l1
                       norm of the y components.
    """
    def __init__(
            self,
            kernel,
            dim_x: int,
            dim_y: int,
            x_train: bool = True,
            y_train: bool = True,
            w_train: bool = True,
            n_comp: int = 0, 
            l1_x: float = 0.,
            l1_y: float = 0.,
            l1_act: float = 0.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.l1_x = l1_x
        self.l1_y = l1_y
        self.l1_act = l1_act

    def build(self, input_shape):
        if (input_shape[1] and input_shape[2] != self.dim_x + 1 
            or len(input_shape) != 3):
            raise ValueError(
                f'Input dimension must be (batch_size, m, {self.dim_x + 1} )'
                f' but it is {input_shape}'
                )
        self.c_x = self.add_weight(
            "c_x",
            shape=(self.n_comp, self.dim_x),
            #initializer=tf.keras.initializers.orthogonal(),
            initializer=tf.keras.initializers.random_normal(),
            trainable=self.x_train)
        self.c_y = self.add_weight(
            "c_y",
            shape=(self.n_comp, self.dim_y),
            #initializer=tf.keras.initializers.orthogonal(),
            #initializer=tf.keras.initializers.random_normal(),
            initializer=tf.keras.initializers.constant(1./self.dim_y),
            trainable=self.y_train)
        self.comp_w = self.add_weight(
            "comp_w",
            shape=(self.n_comp,),
            initializer=tf.keras.initializers.constant(1./self.n_comp),
            trainable=self.w_train) 
        self.eps = 1e-10
        self.built = True

    def call(self, inputs):
        
        # Weight regularizers
        if self.l1_x != 0:
            self.add_loss(self.l1_x * l1_loss(self.c_x))
        if self.l1_y != 0:
            self.add_loss(self.l1_y * l1_loss(self.c_y))
        comp_w = tf.nn.softmax(self.comp_w)
        in_w = inputs[:, :, 0]  # shape (b, n_comp_in)
        in_v = inputs[:, :, 1:] # shape (b, n_comp_in, dim_x)
        out_vw = self.kernel(in_v, self.c_x)  # shape (b, n_comp_in, n_comp)
        out_w = (tf.expand_dims(tf.expand_dims(comp_w, axis=0), axis=0) *
                 tf.square(out_vw)) # shape (b, n_comp_in, n_comp)
        out_w = tf.maximum(out_w, self.eps) #########
        # out_w_sum = tf.maximum(tf.reduce_sum(out_w, axis=2), self.eps)  # shape (b, n_comp_in)
        out_w_sum = tf.reduce_sum(out_w, axis=2) # shape (b, n_comp_in)
        out_w = out_w / tf.expand_dims(out_w_sum, axis=2)
        out_w = tf.einsum('...i,...ij->...j', in_w, out_w, optimize="optimal")
                # shape (b, n_comp)
        if self.l1_act != 0:
            self.add_loss(self.l1_act * l1_loss(out_w))
        out_w = tf.expand_dims(out_w, axis=-1) # shape (b, n_comp, 1)
        out_y_shape = tf.shape(out_w) + tf.constant([0, 0, self.dim_y - 1])
        out_y = tf.broadcast_to(tf.expand_dims(self.c_y, axis=0), out_y_shape)
        out = tf.concat((out_w, out_y), 2)
        return out

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "w_train": self.w_train,
            "l1_x": self.l1_x,
            "l1_y": self.l1_y,
            "l1_act": self.l1_act,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y + 1, self.n_comp)


class KQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp,
                 x_train=True):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.sigma = tf.Variable(0.1, dtype=tf.float32, trainable=True)
        kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train)

    def call(self, inputs):
        rho_x = pure2dm(inputs)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        return probs

class BagKQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp,
                 items,
                 x_train=True,
                 l1_y=0.):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.items = items
        self.sigma = tf.Variable(0.1, dtype=tf.float32, trainable=True)
        kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train,
                            l1_y=l1_y)

    def call(self, inputs):
        bag_size = tf.cast(tf.shape(inputs)[1], tf.float32)
        w = tf.ones_like(inputs[:, :, 0]) / bag_size
        rho_x = comp2dm(w, inputs)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        #rho_y = comp2dm(y_w, y_v)
        return probs


@tf.function
def overlap_kernel(A, B):
    '''
    Calculates the identity kernel between 
    two set of vectors.
    Input:
        A: tensor of shape (bs, d)
        B: tensor of shape (bs, n, d)
    Result:
        K: tensor of shape (bs, n)
    '''
    K = tf.einsum("...d,...nd->...n", A, B)
    return K

def overlap_loss(y_true, y_pred):
    y_true = tf.math.sqrt(y_true)
    overlap = pure_dm_overlap(y_true, y_pred, overlap_kernel)
    #return -tf.reduce_mean(tf.math.log(overlap + 0.0000001), axis=-1) 
    return -tf.reduce_mean(overlap , axis=-1)
 


from sklearn.preprocessing import StandardScaler
import dill as pickle
import numpy as np

def process_data_file(data_path):
    with open(data_path, 'rb') as data_file:
        data_dict = pickle.load(data_file)
    train_x = np.array(data_dict['train_X'])
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(np.array(data_dict['test_X']))
    train_x = train_x[np.argsort(data_dict['bag_id'])]
    num_samples = train_x.shape[0]
    num_cols = train_x.shape[1]
    num_bags = len(data_dict['prop_dict'])
    train_y = np.array([data_dict['prop_dict'][i] for i in range(num_bags)])
    train_y = np.c_[1. - train_y, train_y]
    test_x = test_x[:, np.newaxis, :]
    train_x = np.reshape(train_x, 
                         (num_bags, -1, num_cols))
    test_y = (np.array(data_dict['test_y']) + 1.) / 2.
    test_y = np.c_[1. - test_y, test_y]
    return train_x, train_y, test_x, test_y

"""## Conventional classification experiment"""

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def run_experiment(data_path, data_file, 
                   sigma_prop=1., 
                   n_comp=64, 
                   epochs=3000, 
                   batch_size=128,
                   learning_rate=0.001):

    train_x, train_y, test_x, test_y = process_data_file(data_path + data_file)

    dim_x = train_x.shape[2]
    items_per_bag = train_x.shape[1]
    dim_y = 2
    bgqmc = BagKQMClassModel(dim_x=dim_x, dim_y=dim_y, 
                        n_comp=n_comp, items=items_per_bag)
    bgqmc.predict(train_x[0:1])
    idx = np.random.randint(train_x.shape[0], size=n_comp)
    bgqmc.kqmu.c_x.assign(train_x[idx][:, 0])
    distances = pairwise_distances(train_x[idx][:, 0])
    mean = np.mean(distances)
    std = np.std(distances)
    sigma = mean * sigma_prop
    bgqmc.sigma.assign(sigma)
    print("---------------------------------------new sigma---------------")
    bgqmc.sigma.assign(sigma_prop)
    print(f"Mean: {mean} std:{std} sigma: {sigma} ")
    bgqmc.kqmu.c_y.assign(train_y[idx])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bgqmc.compile(optimizer=optimizer, loss='mse')

    # Fit model
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=50, 
        verbose=1, 
        mode='auto',
        start_from_epoch=300)

    history = bgqmc.fit(train_x, train_y, validation_split=0.1, epochs=epochs, verbose=0,
              batch_size=batch_size,
              callbacks=[model_checkpoint_callback, 
                         early_stopping, 
                         ])
     
    checkpoint_path = "../../models/lpp-{data_path}-{data_file}-{n_comp}-{learning_rate}.ckpt"
    if(not use_stored_model):
        bgqmc.save_weights(checkpoint_path.format(data_path=data_path, data_file=data_file, n_comp=n_comp, learning_rate=learning_rate))
    else:
        bgqmc.load_weights(checkpoint_path.format(data_path=data_path, data_file=data_file, n_comp=n_comp, learning_rate=learning_rate))



    #bgqmc.load_weights(checkpoint_filepath)
    y_pred = bgqmc.predict(test_x)
    accuracy = accuracy_score(np.argmax(test_y, axis=1), 
                              np.argmax(y_pred, axis=1))
    auc = roc_auc_score(test_y[:, 1], y_pred[:, 1] )
    final_sigma = bgqmc.sigma.numpy()
    print(f"data_path:{data_path} data_file:{data_file}  n_comp:{n_comp}  learning_rate:{learning_rate}")
    print(f"accuracy:{accuracy} auc:{auc} final sigma:{final_sigma}")
    
    # Log experiment metrics and configuration
    config ={
        'accuracy': accuracy,
        'auc': auc,
        'final_sigma': final_sigma
    }

    return history, bgqmc, accuracy, auc

def run_hyperparameter_search():
    data_path = "../../Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult1_8_0"

    for n_comp in [4,8,16,32,64,128,256,512,1024,2048]:
        for learning_rate in [0.001, 0.005, 0.0001]:
            for exper in [0, 1]:
                for bag_size in [8, 32, 128, 512]:
                    for rep in range(5):
                        data_file = f"adult{exper}_{bag_size}_{rep}"
                        history, model, acc, auc = run_experiment(data_path, 
                                                        data_file,
                                                        epochs=3000,
                                                        n_comp=n_comp,
                                                        learning_rate=learning_rate)



if use_best_config == False:
    run_hyperparameter_search()
else:

    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult0_8_0"
    n_comp = 32
    learning_rate = 0.005

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)


    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult0_32_0"
    n_comp = 16
    learning_rate = 0.001

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)



    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult0_128_0"
    n_comp = 512
    learning_rate = 0.001

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)

    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult0_512_0"
    n_comp = 64
    learning_rate = 0.005

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)


    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult1_8_0"
    n_comp = 16
    learning_rate = 0.005

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)


    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult1_32_0"
    n_comp = 64
    learning_rate = 0.001

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)



    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult1_128_0"
    n_comp = 128
    learning_rate = 0.001

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)

    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "adult1_512_0"
    n_comp = 64
    learning_rate = 0.005

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)



    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic0_8_4"
    n_comp = 256
    learning_rate = 0.005
    sigma = 1.242 * 10

    run_experiment(data_path, data_file, sigma_prop=sigma, epochs=2000, n_comp=n_comp, learning_rate=learning_rate)


    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic0_32_3"
    n_comp = 128
    learning_rate = 0.005
    sigma = 1.116*9

    run_experiment(data_path, data_file,sigma_prop=sigma, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)



    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic0_128_2"
    n_comp = 128
    learning_rate = 0.001
    sigma = 1.116* 4

    run_experiment(data_path, data_file, sigma_prop=sigma, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)

    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic0_512_3"
    n_comp = 256
    learning_rate = 0.005
    sigma = 1.485* 10 #0.867
    #sigma = 1.485* 5

    run_experiment(data_path, data_file, sigma_prop=sigma, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)


    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic1_8_0"
    n_comp = 16
    learning_rate = 0.005

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)


    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic1_32_0"
    n_comp = 128
    learning_rate = 0.001

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)



    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic1_128_0"
    n_comp = 32
    learning_rate = 0.001

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)

    data_path = "../../scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/"
    data_file = "magic1_512_0"
    n_comp = 128
    learning_rate = 0.005

    run_experiment(data_path, data_file, epochs=3000, n_comp=n_comp, learning_rate=learning_rate)


