import tensorflow as tf
from tensorflow import keras

def create_model(X):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X.shape[1])),
        #keras.layers.BatchNormalization(),
        #keras.layers.Dropout(.25),
        #keras.layers.Dense(512, activation="sigmoid"),
        keras.layers.Dense(128, activation="sigmoid", ),
        #keras.layers.Dense(256, activation="relu", ),
        #keras.layers.Dense(512, activation="relu", ),
        #keras.layers.Dense(1024, activation="relu", ),
        #keras.layers.Dense(1024, activation="relu", ),
        #keras.layers.Dense(512, activation="relu",),
        #keras.layers.Dense(256, activation="relu",),
        #keras.layers.Dropout(.25),
        keras.layers.Dense(128, activation="relu", ),
        #keras.layers.Dropout(.25),
        keras.layers.Dense(64, activation="relu", ),
        keras.layers.Dense(1, )#activation="sigmoid")
    ])
    
    return model