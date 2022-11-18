import tensorflow as tf
from tensorflow import keras

def create_model(X):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X.shape[1])),
        #keras.layers.BatchNormalization(),
        #keras.layers.Dropout(.25),
        #keras.layers.Dense(512, activation="sigmoid"),
        keras.layers.Dense(128, activation="sigmoid", ),
        keras.layers.Dense(256, activation="relu", ),
        #keras.layers.Dense(256, activation=ACTIVATION, kernel_regularizer="l2"),
        #keras.layers.Dense(512, activation=ACTIVATION, kernel_regularizer="l2"),
        #keras.layers.Dropout(.25),
        keras.layers.Dense(128, activation="relu",),
        #keras.layers.Dropout(.25),
        keras.layers.Dense(32, activation="relu",),
        keras.layers.Dense(1)
    ])
    
    return model