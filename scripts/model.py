from comet_ml import Experiment

from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.objectives import binary_crossentropy
from keras.optimizers import Adam

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D




class VAE():

    def __init__(self,
                 input_shape,
                 latent_dim = 3,
                 name = "mnist",
                 lr = .02,
                 filters = 4,
                 kernel_size = 3,
                 std = 2):
        self.name = name
        self.experiment = Experiment(
            #api_key="YOUR API KEY",
            # or
            api_key="Zu2hmmkc0SbxvsPTii51xL38z",
            project_name=self.name)
        self.experiment.log_parameters(
            { " latent_dim" : latent_dim,
              "lr" : lr,
              "filters" : filters,
              "kernel_size" : kernel_size,
              "std" : std
            }
        )

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            filters *= 2
            x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    strides=2,
                    padding='same')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)

        #sampling
        def _sampling(args):
            z_mean = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + float(std) * epsilon

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(_sampling, output_shape=(latent_dim,), name='z')(z_mean)

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(2):
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                kernel_size=kernel_size,
                                activation='sigmoid',
                                padding='same',
                                name='decoder_output')(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[1])
        self.vae = Model(inputs, outputs, name='vae')

        # Compile the autoencoder computation graph
        self.vae.compile(optimizer=Adam(lr = lr), loss="binary_crossentropy", metrics=["mae"])

    def train(self,
              x_train,
              epochs =100,
              batch_size = 512,
              val_p = .1):
        filepath="../weights/best_" + self.name + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.vae.fit(x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks = [checkpoint],
                validation_split = val_p,
                shuffle = True)
        return self.vae

    def reconstruct(self,
                x_test):
        return self.decoder.predict(self.encode(x_test))

    def encode(self,
               x_test):
        return self.encoder.predict(x_test)[1]

    def plot_encoding(self,
                     x,
                     y):
        x_encoding = self.encode(x)
        plt.scatter(x_encoding[:, 0], x_encoding[:, 1], c = y[:, 0], marker = 'o')
        plt.colorbar()


    def load_weights(self, path):
        self.vae.load_weights(path)

    def analogy(self, a, b, A):
        plt.imshow(a[:, :, 0])
        plt.show()
        print('Is to')
        plt.imshow(b[:, :, 0])
        plt.show()
        print('As')
        plt.imshow(A[:, :, 0])
        plt.show()
        print("Is to")
        attribute_vector = self.encode(np.expand_dims(b, 0)) - self.encode(np.expand_dims(a, 0))
        latent_embedding = attribute_vector + self.encode(np.expand_dims(A, 0))
        plt.imshow(self.decoder.predict(latent_embedding).reshape(A.shape)[:, :, 0])
        plt.show()
