from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Activation, BatchNormalization, Dropout, Flatten, Dense, Concatenate

def create_autoencoder(encoder, decoder):
    input_shape = encoder.input_shape[1:]
    inputs = Input(shape=input_shape)
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = Model(inputs, decoded)
    return autoencoder


#def create_block(input, chs): ## Convolution block of 2 layers
#    x = input
#    for i in range(2):
#        x = Conv2D(chs, 3, padding="same")(x)
#        x = Activation("relu")(x)
#        x = BatchNormalization()(x)
#    return x

#def end_to_end():  ## I commented several layers of the model for descreasing model complexity as the results were almost same
#    input = Input((32,32,3))
#    
#    # Encoder
#    block1 = create_block(input, 32)
#    x = MaxPool2D(2)(block1)
#    block2 = create_block(x, 64)
#    x = MaxPool2D(2)(block2)
#    #block3 = create_block(x, 64)
#    #x = MaxPool2D(2)(block3)
#    #block4 = create_block(x, 128)
#    
#    # Middle
#    #x = MaxPool2D(2)(block2)
#    middle = create_block(x, 128)
#    
#    # Decoder
#    #x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)
#    #x = Concatenate()([block4, x])
#    #x = create_block(x, 128)
#    #x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
#    #x = Concatenate()([block3, x])
#    #x = create_block(x, 64)
#    x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)
#    x = Concatenate()([block2, x])
#    x = create_block(x, 64)
#    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)
#    x = Concatenate()([block1, x])
#    x = create_block(x, 32)
#    
#    # reconstruction
#    decoder = Conv2D(3, 1)(x)
#    recon = Activation("sigmoid", name='autoencoder')(decoder)
#    
#    #classification 
#    c = Conv2D(1024, 3, padding="same")(middle)
#    c = Activation('relu')(c)
#    c = BatchNormalization()(c)
#    c = MaxPool2D(2)(c)
#    c = Dropout(0.5)(c)
#    c = Conv2D(128, 3, padding="same")(c)
#    c = Activation('relu')(c)
#    c = BatchNormalization()(c)
#    c = MaxPool2D(2)(c)
#    c = Dropout(0.4)(c)
#    c = Flatten()(c)
#    c = Dense(512, activation='relu')(c)
#    c = Dropout(0.35)(c)
#    c = Dense(100, activation='relu')(c)
#    c = Dropout(0.69)(c)
#    classify = Dense(10, activation='softmax', name='classification')(c)
#    
#    outputs = [recon, classify]
#    
#    return Model(input, outputs), Model(input, middle), Model(middle, middledecoder)




#def end_to_end():  ## I commented several layers of the model for descreasing model complexity as the results were almost same
#    input = Input((32,32,3))
#    
#    # Encoder
#    block1 = create_block(input, 32)
#    x = MaxPool2D(2)(block1)
#    block2 = create_block(x, 64)
#    x = MaxPool2D(2)(block2)
#    #block3 = create_block(x, 64)
#    #x = MaxPool2D(2)(block3)
#    #block4 = create_block(x, 128)
#    
#    # Middle
#    #x = MaxPool2D(2)(block2)
#    middle = create_block(x, 128)
#    
#    # Decoder
#    #x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)
#    #x = Concatenate()([block4, x])
#    #x = create_block(x, 128)
#    #x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
#    #x = Concatenate()([block3, x])
#    #x = create_block(x, 64)
#    x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)
#    x = Concatenate()([block2, x])
#    x = create_block(x, 64)
#    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)
#    x = Concatenate()([block1, x])
#    x = create_block(x, 32)
#    
#    # reconstruction
#    x = Conv2D(3, 1)(x)
#    recon = Activation("sigmoid", name='autoencoder')(x)
#    
#    #classification 
#    c = Conv2D(1024, 3, padding="same")(middle)
#    c = Activation('relu')(c)
#    c = BatchNormalization()(c)
#    c = MaxPool2D(2)(c)
#    c = Dropout(0.5)(c)
#    c = Conv2D(128, 3, padding="same")(c)
#    c = Activation('relu')(c)
#    c = BatchNormalization()(c)
#    c = MaxPool2D(2)(c)
#    c = Dropout(0.4)(c)
#    c = Flatten()(c)
#    c = Dense(512, activation='relu')(c)
#    c = Dropout(0.35)(c)
#    c = Dense(100, activation='relu')(c)
#    c = Dropout(0.69)(c)
#    classify = Dense(10, activation='softmax', name='classification')(c)
#    
#    outputs = [recon, classify]
#    
#    return Model(input, outputs)
#
