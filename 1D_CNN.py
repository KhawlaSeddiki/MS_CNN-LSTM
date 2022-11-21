#!/usr/bin/env python3
# --- coding: utf-8 ---

import numpy as np
import os
import pandas as pd
import keras
from tensorflow.keras.models import Model
from keras.utils import np_utils
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D
from keras.optimizer_v2 import adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import minmax_scale

chunk = pd.read_csv('input_filename.csv', header=None, chunksize=10000)
df = pd.concat(chunk)

##### Data split into train and test
def split_dataset(data):
    split = int(0.8 * len(df))
    train_x = np.array(df[:split])
    test_x = np.array(df[split:])
    # reshape train and test
    train_sp = minmax_scale(train_x[:, 1:], axis=0, feature_range=(0, 5))
    train = np.reshape(train_sp, (train_sp.shape[0], train_sp.shape[1], 1))
    y_train = np_utils.to_categorical(train_x[:, 0])
    test_sp = minmax_scale(test_x[:, 1:], axis=0, feature_range=(0, 5))
    test = np.reshape(test_sp, (test_sp.shape[0], test_sp.shape[1], 1))
    y_test = np_utils.to_categorical(test_x[:, 0])
    return train, y_train, test, y_test


red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1, min_lr=0.000000001)
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto', restore_best_weights=True)
train, y_train, test, y_test = split_dataset(df)
my_filters = 32
my_kernel_size = 21

##### 1D CNN model
def build_model():
    model = Sequential([
        Conv1D(filters=my_filters, kernel_size=my_kernel_size, strides=1, padding='same', activation='relu',
               kernel_initializer=keras.initializers.he_normal(), input_shape= (train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(1000, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='sigmoid')
    ])
    # compile the model and fit it
    model.compile(loss='binary_crossentropy', optimizer=adam.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
    model.fit(train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[earlyStop,red_lr], verbose=1)
    # evaluate model and get accuracy
    _, accuracy = model.evaluate(test, y_test, verbose=1)
    accuracy = accuracy * 100.0
    print('Accuracy of Model: ', accuracy)
    # save the model
    # model.save("trained_model")
    return model

def check_GPUs():
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Calling main function
if __name__=="__main__":
  check_GPUs()
  main()
