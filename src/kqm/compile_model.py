import tensorflow as tf
from keras import losses
from keras import metrics
from keras import optimizers

def compile_model(model, adam_lr):
    model.compile(optimizer=optimizers.Adam(learning_rate=adam_lr),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


