# Title: 'train.py'
# Author: Curcuraci L.
# Date: 20/10/2022
#
# Scope: Feature extraction+clustering+reconstruction in patch space

"""

"""

#################
#####   LIBRARIES
#################


import numpy as np
import matplotlib.pyplot as plt
import os
import random

from datetime import datetime
from bmmltools.utils.basic import manage_path,standard_number

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Dropout, ActivityRegularization, \
    Conv3D, Conv3DTranspose, MaxPool3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


#############
#####   CLASS
#############


class ConvAutoencoder2D:

    def __init__(self,input_shape,use_tensorboard=True,denoising=False,p_dropout=0.2,sparsify=False,l1_strenght=1e-4):

        self.seed = None
        self.input_shape = input_shape
        self.denoising = denoising
        self.p_dropout = p_dropout
        self.sparsify = sparsify
        self.l1_strenght = l1_strenght
        self.use_tensorboard = use_tensorboard
        self.model = self.build_autoencoder(input_shape)

        # set the random state
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def encoder(self,input_shape):

        In = Input(input_shape)

        x = In
        if self.denoising:

            x = Dropout(self.p)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2), padding='same')(x)
        if self.denoising:

            x = Dropout(self.p)(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPool2D((2, 2), padding='same')(x)
        if self.denoising:

            x = Dropout(self.p)(x)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x= MaxPool2D((2, 2), padding='same')(x)
        if self.sparsify:

            x = ActivityRegularization(l1=self.l1_strenght)(x)

        Out = x
        return Model(In, Out, name='encoder')

    def decoder(self,input_shape):

        In = Input(input_shape)
        x = Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(In)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        Out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        return Model(In, Out, name='decoder')

    def build_autoencoder(self,input_shape):

        # initialize encoder and decoder
        en = self.encoder(input_shape)
        bottleneck_shape = en(np.ones((1,) + input_shape)).shape[1:]
        dec = self.decoder(bottleneck_shape)

        # compose the autoencoder
        In = Input(input_shape)
        x = en(In)
        Out = dec(x)
        return Model(In, Out)

    def compile_model(self,opt='adam',loss='binary_crossentropy',metrics=['mse']):

        self.model.compile(optimizer=opt, loss=loss,metrics=metrics)

    def train_model(self,X,epochs=50,batch_size=32,test_faction=0.1):

        if test_faction>0:

            split_idx = int(len(X)*test_faction)
            random.shuffle(X)
            Xtest = X[:split_idx]
            Xtrain = X[split_idx:]
            validation_data = (Xtest,Xtest)

        else:

            Xtrain = X
            validation_data = None

        print('>----------------<')
        print('Start training...')
        callbacks_list = None
        if self.use_tensorboard:

            callbacks_list = [tf.keras.callbacks.TensorBoard(
                                log_dir = os.getcwd()+os.sep+'ConvAutoencoder/tensorboard/logs/fit/'
                                            +datetime.now().strftime("%Y%m%d-%H%M%S"),
                                histogram_freq=1)]

        info = self.model.fit(x=Xtrain,y=Xtrain, epochs=epochs, batch_size=batch_size, shuffle=True,
                              validation_data = validation_data, callbacks=callbacks_list)
        print('\n...autoencoder trained!')
        print('>----------------<')
        return info

    def inference(self,X):

        return self.model.predict(X,verbose=0)

    def save_model(self,saving_folder_path):

        print('>----------------<')
        model_code = standard_number(np.random.randint(0,9999),4)
        model_saving_path = manage_path(saving_folder_path)+os.sep+'conv_autoencoder_{}.h5'.format(model_code)
        self.model.save(model_saving_path)
        print('Autoencoder model saved at {}'.format(model_saving_path))

        weights_model_saving_path = manage_path(saving_folder_path) + os.sep + \
                                    'conv_autoencoder_{}_weights.h5'.format(model_code)
        self.model.save_weights(weights_model_saving_path)
        print('Autoencoder model weights saved at {}'.format(weights_model_saving_path))

        encoder = self.extract_encoder()
        encoder_saving_path = manage_path(saving_folder_path) + os.sep + 'encoder_{}.h5'.format(model_code)
        encoder.save(encoder_saving_path)
        print('Encoder saved at {}'.format(encoder_saving_path))
        weights_encoder_saving_path =  manage_path(saving_folder_path) + os.sep + \
                                    'encoder_{}_weights.h5'.format(model_code)
        encoder.save_weights(weights_encoder_saving_path)
        print('Encoder weights saved at {}'.format(weights_encoder_saving_path))

        decoder = self.extract_decoder()
        decoder_saving_path = manage_path(saving_folder_path) + os.sep + 'decoder_{}.h5'.format(model_code)
        decoder.save(decoder_saving_path)
        print('Decoder model saved at {}'.format(decoder_saving_path))
        weights_decoder_saving_path =  manage_path(saving_folder_path) + os.sep + \
                                    'decoder_{}_weights.h5'.format(model_code)
        decoder.save_weights(weights_decoder_saving_path)
        print('Decoder weights saved at {}'.format(weights_decoder_saving_path))
        print('>----------------<')

    def extract_encoder(self):

        return self.model.get_layer('encoder')

    def extract_decoder(self):

        return self.model.get_layer('decoder')

class ConvAutoencoder3D:

    def __init__(self,input_shape,use_tensorboard=True,denoising=False,p_dropout=0.2,sparsify=False,l1_strenght=1e-4):

        self.seed = None
        self.input_shape = input_shape
        self.denoising = denoising
        self.p_dropout = p_dropout
        self.sparsify = sparsify
        self.l1_strenght = l1_strenght
        self.use_tensorboard = use_tensorboard
        self.model = self.build_autoencoder(input_shape)

        # set the random state
        random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def encoder(self,input_shape):

        In = Input(input_shape)

        x = In
        if self.denoising:

            x = Dropout(self.p)(x)

        x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPool3D((2, 2, 2), padding='same')(x)
        if self.denoising:

            x = Dropout(self.p)(x)

        x = Conv2D(32, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPool3D((2, 2, 2), padding='same')(x)
        if self.denoising:

            x = Dropout(self.p)(x)

        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same',kernel_regularizer = regularizers.L1(1e-4))(x)
        x= MaxPool3D((2, 2, 2), padding='same')(x)
        if self.sparsify:

            x = ActivityRegularization(l1=self.l1_strenght)(x)

        Out = x
        return Model(In, Out, name='encoder')

    def decoder(self,input_shape):

        In = Input(input_shape)
        x = Conv3DTranspose(16, (3, 3, 3), strides=2, activation='relu', padding='same')(In)
        x = Conv3DTranspose(32, (3, 3, 3), strides=2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(64, (3, 3, 3), strides=2, activation='relu', padding='same')(x)
        Out = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)
        return Model(In, Out, name='decoder')

    def build_autoencoder(self,input_shape):

        # initialize encoder and decoder
        en = self.encoder(input_shape)
        bottleneck_shape = en(np.ones((1,) + input_shape)).shape[1:]
        dec = self.decoder(bottleneck_shape)

        # compose the autoencoder
        In = Input(input_shape)
        x = en(In)
        Out = dec(x)
        return Model(In, Out)

    def compile_model(self,opt='adam',loss='binary_crossentropy',metrics=['mse']):

        self.model.compile(optimizer=opt, loss=loss,metrics=metrics)

    def train_model(self,X,epochs=50,batch_size=32,test_faction=0.1):

        if test_faction>0:

            split_idx = int(len(X)*test_faction)
            random.shuffle(X)
            Xtest = X[:split_idx]
            Xtrain = X[split_idx:]
            validation_data = (Xtest,Xtest)

        else:

            Xtrain = X
            validation_data = None

        print('>----------------<')
        print('Start training...')
        callbacks_list = None
        if self.use_tensorboard:

            callbacks_list = [tf.keras.callbacks.TensorBoard(
                                log_dir = os.getcwd()+os.sep+'ConvAutoencoder/tensorboard/logs/fit/'
                                            +datetime.now().strftime("%Y%m%d-%H%M%S"),
                                histogram_freq=1)]

        info = self.model.fit(x=Xtrain,y=Xtrain, epochs=epochs, batch_size=batch_size, shuffle=True,
                              validation_data = validation_data, callbacks=callbacks_list)
        print('\n...autoencoder trained!')
        print('>----------------<')
        return info

    def inference(self,X):

        return self.model.predict(X,verbose=0)

    def save_model(self,saving_folder_path):

        print('>----------------<')
        model_code = standard_number(np.random.randint(0,9999),4)
        model_saving_path = manage_path(saving_folder_path)+os.sep+'conv_autoencoder_{}.h5'.format(model_code)
        self.model.save(model_saving_path)
        print('Autoencoder model saved at {}'.format(model_saving_path))

        weights_model_saving_path = manage_path(saving_folder_path) + os.sep + \
                                    'conv_autoencoder_{}_weights.h5'.format(model_code)
        self.model.save_weights(weights_model_saving_path)
        print('Autoencoder model weights saved at {}'.format(weights_model_saving_path))

        encoder = self.extract_encoder()
        encoder_saving_path = manage_path(saving_folder_path) + os.sep + 'encoder_{}.h5'.format(model_code)
        encoder.save(encoder_saving_path)
        print('Encoder saved at {}'.format(encoder_saving_path))
        weights_encoder_saving_path =  manage_path(saving_folder_path) + os.sep + \
                                    'encoder_{}_weights.h5'.format(model_code)
        encoder.save_weights(weights_encoder_saving_path)
        print('Encoder weights saved at {}'.format(weights_encoder_saving_path))

        decoder = self.extract_decoder()
        decoder_saving_path = manage_path(saving_folder_path) + os.sep + 'decoder_{}.h5'.format(model_code)
        decoder.save(decoder_saving_path)
        print('Decoder model saved at {}'.format(decoder_saving_path))
        weights_decoder_saving_path =  manage_path(saving_folder_path) + os.sep + \
                                    'decoder_{}_weights.h5'.format(model_code)
        decoder.save_weights(weights_decoder_saving_path)
        print('Decoder weights saved at {}'.format(weights_decoder_saving_path))
        print('>----------------<')

    def extract_encoder(self):

        return self.model.get_layer('encoder')

    def extract_decoder(self):

        return self.model.get_layer('decoder')



############
#####   MAIN
############


### inputs

# path
path_to_dataset = r'dev/dataset/dataset.npy'
model_saving_folder_path = r'dev/models'

# train setting
n_epochs = 300
batch_size = 32

### setup

# load dataset
X = np.load(path_to_dataset)


### Model

# train autoencoder 2D
# P.A.: dataset shape convention: (batch_dim, y_dim, x_dim, channel_dim)
autoencoder = ConvAutoencoder2D((200,200,1),sparsify=False,denoising=False)     # network input shape to specify: (y_dim, x_dim, channel_dim)
autoencoder.compile_model()
autoencoder.train_model(X[:,:,:,None],epochs=n_epochs,batch_size=batch_size)     # modify here based on the X shape
autoencoder.save_model(model_saving_folder_path)

# # train autoencoder 3D
# # P.A.:dataset shape convention: (batch_dim, z_dim, y_dim, x_dim, channel_dim)
# autoencoder = ConvAutoencoder2D((200,200,200,1),sparsify=False,denoising=False)    # network shape to specify: (z_dim, y_dim, x_dim, channel_dim)
# autoencoder.compile_model()
# autoencoder.train_model(X[:,:,:,:,None],epochs=n_epochs,batch_size=batch_size)    # modify here based on the X shape
# autoencoder.save_model(model_saving_folder_path)