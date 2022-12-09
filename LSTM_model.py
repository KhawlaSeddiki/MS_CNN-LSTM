import numpy as np
import os
import glob
import pandas as pd
import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LSTM
from keras import regularizers
from keras.utils import np_utils
from keras.optimizer_v2 import adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import minmax_scale


# inport data
path = 'path'
my_files = sorted(glob.glob(path + "*.csv"))
data = list()
for f in my_files:
    df = pd.read_csv(f)
    values = df.values
    data.append(values)
    print(f)

##### Data split into train and test
def split_dataset(data):
    split = int(0.8 * len(data))
    train_x = np.array(data[:split])
    test_x = np.array(data[split:])
    # reshape train and test
    train_sp = minmax_scale(train_x[:, 1:], axis=0, feature_range=(0, 1))
    train = np.reshape(train_sp, (train_sp.shape[0], train_sp.shape[1], 1))
    y_train = np_utils.to_categorical(train_x[:, 0])
    test_sp = minmax_scale(test_x[:, 1:], axis=0, feature_range=(0, 1))
    test = np.reshape(test_sp, (test_sp.shape[0], test_sp.shape[1], 1))
    y_test = np_utils.to_categorical(test_x[:, 0])
    return train, y_train, test, y_test


train, y_train, test, y_test = split_dataset(df)
nb_neurons = 30
my_filters = 32
my_kernel_size = 21

##### LSTM model
def LSTM_model():
    red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1, min_lr=0.000000001)
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto', restore_best_weights=True)
    model = Sequential()
    model.add(LSTM(nb_neurons, activation='tanh', input_shape=(train.shape[1], train.shape[2]))) # ,
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model and fit it
    model.compile(loss='binary_crossentropy', optimizer=adam.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
    model.fit(train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[earlyStop, red_lr], verbose=1)
    # evaluate model and get accuracy
    _, accuracy = model.evaluate(test, y_test, verbose=1)
    accuracy = accuracy * 100.0
    print('Accuracy of Model: ', accuracy)
    # save the model
    # model.save("LSTM_model.h5")
    return model


##### CNN-LSTM model
def CNNLSTM_model():
    red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1, min_lr=0.000000001)
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto', restore_best_weights=True)
    model = Sequential()
    model.add(Conv1D(filters=my_filters, kernel_size=my_kernel_size, padding="same", activation='relu', input_shape=(train.shape[1], train.shape[2])))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, padding="same")) ,
    model.add(LSTM(units=nb_neurons, activation='tanh', return_sequences=True, recurrent_dropout=0.5)) 
    model.add(LSTM(units=nb_neurons, activation='tanh', recurrent_dropout=0.5))
    model.add(Dense(units=1, activation="sigmoid"))
    # compile the model and fit it
    model.compile(loss='binary_crossentropy', optimizer=adam.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
    model.fit(train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[earlyStop, red_lr], verbose=1)
    # evaluate model and get accuracy
    _, accuracy = model.evaluate(test, y_test, verbose=1)
    accuracy = accuracy * 100.0
    print('Accuracy of Model: ', accuracy)
    # save the model
    # model.save("CNN_LSTM_model.h5")
    return model
