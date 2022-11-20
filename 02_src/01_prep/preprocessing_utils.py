import pandas as pd
import numpy as np

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

        
def categorical_encoding_and_scaling(x, categorical, numeric_mean_std, numeric_max):
    '''
        One-hot-encoding of categorical data, and scaling numeric values.
        Parameters:
         - x: dataset as pandas.DataFrame
         - categorical: list of categorical column names (list of strings)
         - numeric: list of numerical column names (list of strings)
        Returns:
         - a numpy.Array containing the transformed data
    '''
    #categorical encoding:
    cat_coded = None
    for category in categorical:
        #integer category encoding:
        codes = x[category].unique()
        code_dictionary = {}
        values = []
        for i,code in enumerate(codes):
            code_dictionary[code] = i
        for _,r in x.iterrows():
            values.append(code_dictionary[r[category]])

        #one-hot-conding:
        encoder = tf.keras.layers.CategoryEncoding(num_tokens=len(code_dictionary),
                                                 output_mode="one_hot")
        cat_coded = encoder(values) if cat_coded is None else np.hstack([cat_coded, encoder(values)])
        
    #-mean/std numerical encoding:
    num_scaled = None
    for num in numeric_mean_std:
        scaler = tf.keras.layers.Normalization(axis=None)
        scaler.adapt(x[num])
        num_scaled = scaler(x[num]).numpy().T if num_scaled is None else np.hstack([num_scaled, scaler(x[num]).numpy().T])
    #/max encoding:
    for num in numeric_max:
        print((x[num]/max(x[num])).shape)
        print(num_scaled.shape)
        num_scaled = np.hstack([num_scaled, np.array(x[num]/max(x[num])).reshape((len(num_scaled), 1))])
        
    #creating output:
    return np.hstack([cat_coded, num_scaled]) if not(num_scaled is None) else cat_coded.numpy()

def normalize_series(y):
    scaler = tf.keras.layers.Normalization(axis=None)
    scaler.adapt(y)
    return scaler(y).numpy()