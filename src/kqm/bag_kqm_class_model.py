import tensorflow as tf
from kqm import comp2dm,dm2comp,pure_dm_overlap,RBFKernelLayer,KQMUnit

class BagKQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp,
                 items,
                 x_train=True,
                 l1_y=0.,
                 trainable_sigma=True,
                 min_sigma=1e-3):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.items = items
        self.sigma = tf.Variable(0.1, dtype=tf.float32, trainable=True)
        self.kernel_x = RBFKernelLayer(self.sigma, dim=dim_x, 
                                       trainable=trainable_sigma,
                                       min_sigma=min_sigma)
        self.kqmu = KQMUnit(self.kernel_x,
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
