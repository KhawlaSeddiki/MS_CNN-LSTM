import numpy as np
import os
import glob
import pandas as pd
import keras
from tensorflow.keras.models import Model
from keras.models import Sequential, model_from_json, load_model
from sklearn.preprocessing import MinMaxScaler

# import data
path = 'path'
my_files = sorted(glob.glob(path + "*.csv"))
data = list()
for f in my_files:
    df = pd.read_csv(f)
    values = df.values
    data.append(values)

# import CNN model
base_model = load_model('1DCNN_model.h5')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense').output)
scaler = MinMaxScaler(feature_range=(0, 1))

# data embedding with the dense layer of CNN model
for i in range(len(data)):
    sp_data = scaler.fit_transform(data[i][:, 1:].reshape(len(df), -1)).reshape((len(df), len(df.columns)-1, 1))
    embedding = model.predict(sp_data)
    data = np.column_stack((data[i][:, 0], embedding))
    filename = 'embedded_data' + str(i + 1) + '.csv'
    df.to_csv(filename, index=False)
