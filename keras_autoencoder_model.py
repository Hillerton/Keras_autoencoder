"""
defines one class with two subfunctions one for creating a model in keras and
one to fit data. Exectued in init_autoencoder.py to train model.
Made for keras 2.1.3
"""

import keras
from keras import Sequential
from keras.layers import Input, Reshape, Flatten, Dense, Activation, Dropout, ActivityRegularization
from keras.models import Model, load_model
import numpy as np
from keras.layers import advanced_activations
from keras.models import Model, load_model
import tensorflow as tf

class keras_autoencoder:

    def __init__(self, nr_classes, nr_features, nr_hidden, l1_lambda, noise):
        tot_features = nr_classes * nr_features
        input = Input(shape=(nr_features, nr_classes))

        lyr = Reshape((tot_features,))(input)
        lyr = Dropout(noise)(lyr)
        lyr = Dense(nr_hidden, activation="linear")(lyr)
        lyr = advanced_activations.LeakyReLU()(lyr)

        #lyr = ActivityRegularization(l1_lambda)(lyr)

        lyr = Dense(tot_features, activation="linear")(lyr)
        lyr = Reshape((nr_features, nr_classes))(lyr)
        lyr = Activation("softmax")(lyr)

        model = Model(inputs=input, outputs=lyr)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['categorical_accuracy', 'mae'])
        self.model = model

    def fit_model(self, x_train, x_val, epochs, tb):
        self.model.fit(x_train, x_train, epochs=epochs, batch_size=32, shuffle=True, verbose=2, validation_data=(x_val, x_val), callbacks=[tb])

    def save(self, path):
        self.model.save(path+"/AE_model.h5")
        model_json = self.model.to_json()
        with open(path+"/AE_model.json", "w+") as json_file:
            json_file.write(model_json)
        self.model.save_weights(path+"/AE_weights_model.h5")
