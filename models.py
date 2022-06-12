from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class LSTM():
    def __init__(self, data, seed=5):
        reset_random_seeds(seed)
        self.data = data
        input_lags, no_features = data.lags, data.no_features

        input = keras.Input(shape=(input_lags, no_features), name="input")
        x = layers.LSTM(64)(input)
        pred_price = layers.Dense(1, activation="linear", name="price")(x)

        self.model = keras.Model(
            inputs = input,
            outputs = pred_price
        )

    def fit_compile(self, no_epochs):
        train = self.data.train_generator
        test = self.data.test_generator

        lr_schedule = ExponentialDecay(
            0.0005,
            decay_steps=2*self.data.no_bs,
            decay_rate=0.96,
            staircase=True)

        opt = Adam(learning_rate=lr_schedule)
        
        self.model.compile(
            loss="mean_squared_error", optimizer=opt
        )
        history = self.model.fit(
            train,
            validation_data=test,
            epochs=no_epochs,
            shuffle=False,
            verbose=0
        )

        return history

class BILSTM():
    def __init__(self, data, seed=5):
        reset_random_seeds(seed)
        self.data = data
        input_lags, no_features = data.lags, data.no_features

        input = keras.Input(shape=(input_lags, no_features), name="input")
        x = layers.Bidirectional(layers.LSTM(128))(input)
        pred_price = layers.Dense(1, activation="linear", name="price")(x)

        self.model = keras.Model(
            inputs = input,
            outputs = pred_price
        )

    def fit_compile(self, no_epochs):
        train = self.data.train_generator
        test = self.data.test_generator

        lr_schedule = ExponentialDecay(
            0.0005,
            decay_steps=2*self.data.no_bs,
            decay_rate=0.96,
            staircase=True)


        opt = Adam(learning_rate=lr_schedule)

        self.model.compile(
            loss="mean_squared_error", optimizer=opt
        )
        history = self.model.fit(
            train,
            validation_data=test,
            epochs=no_epochs,
            shuffle=False,
            verbose=0
        )

        return history


class CNN_BILSTM():
    def __init__(self, data,seed=5):
        reset_random_seeds(seed)
        self.data = data
        input_lags, no_features = data.lags, data.no_features

        input = keras.Input(shape=(input_lags, no_features), name="input")
        x = layers.Conv1D(128, kernel_size=1, activation='relu', padding="same")(input)
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dense(16, activation='relu')(x)
        pred_price = layers.Dense(1, activation="linear", name="price")(x)

        self.model = keras.Model(
            inputs = input,
            outputs = pred_price
        )

    def fit_compile(self, no_epochs):
        train = self.data.train_generator
        test = self.data.test_generator

        lr_schedule = ExponentialDecay(
            0.0005,
            decay_steps=2*self.data.no_bs,
            decay_rate=0.96,
            staircase=True)

        opt = Adam(learning_rate=lr_schedule)

        self.model.compile(
            loss="mean_squared_error", optimizer=opt
        )
        history = self.model.fit(
            train,
            validation_data=test,
            epochs=no_epochs,
            shuffle=False,
            verbose=0
        )

        return history

class CNN2PRED():
    def __init__(self, data, seed=5):
        reset_random_seeds(seed)
        self.data = data
        input_lags, no_features = data.lags, data.no_features

        input = keras.Input(shape=(input_lags, no_features), name="input")
        x = layers.Reshape((input_lags, no_features,1))(input)
        x = layers.Conv2D(32, kernel_size=(1, no_features), activation='relu')(x)
        x = layers.Conv2D(32, kernel_size=(3, 1), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)
        x = layers.Conv2D(8, kernel_size=(3, 1), activation='relu')(x)
        x = layers.Flatten()(x)
        pred_price = layers.Dense(1, activation="linear", name="price")(x)

        self.model = keras.Model(
            inputs = input,
            outputs = pred_price
        )

    def fit_compile(self, no_epochs):
        train = self.data.train_generator
        test = self.data.test_generator

        lr_schedule = ExponentialDecay(
            0.0005,
            decay_steps=3*self.data.no_bs,
            decay_rate=0.96,
            staircase=True)

        opt = Adam(learning_rate=lr_schedule)

        self.model.compile(
            loss="mean_squared_error", optimizer=opt
        )
        history = self.model.fit(
            train,
            validation_data=test,
            epochs=no_epochs,
            shuffle=False,
            verbose=0
        )

        return history
