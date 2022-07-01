# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:17:28 2022

@author: jases
"""

#1. Import packages
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd
import matplotlib.pyplot as plt

#2. Read the data from the csv file
file_path= r"C:\Users\jases\Desktop\AI-06\DL\data\garments_worker_productivity.csv"
data= pd.read_csv(file_path, sep=',', header= 0 )

#Replace missing values with 0
data = data.fillna(value=0)

#Drop useless columns
data= data.drop(['date', 'quarter', 'day'], axis= 1)

print(data.isna().sum())

data['actual_productivity'].value_counts()

data = pd.get_dummies(data)

feature= data.copy()
label= data.pop('actual_productivity')


SEED=12345
x_train, x_test, y_train, y_test= train_test_split(feature, label, test_size=0.2, random_state=SEED)


#Perform data normalization
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train= standardizer.transform(x_train)
x_test= standardizer.transform(x_test)

#Build a NN with 3 hidden layers
model= keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

base_log_path = r"C:\Users\jases\Desktop\AI-06\DL\tb_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ '___Project2')
tb = TensorBoard(log_dir=log_path)
batch_size = 16
es= EarlyStopping(monitor='val_loss',patience=5) 

#Compile the model
model.compile(optimizer='adam',loss='mse', metrics=['mae'])

#Train the model
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=batch_size,epochs=100,callbacks=[tb, es])

#Visualize the loss and accuracy
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = history.epoch

plt.plot(epochs,training_loss,label='Training Loss')
plt.plot(epochs,val_loss,label='Validation Loss')
plt.legend()
plt.figure()

plt.show()
test_result = model.evaluate(x_test,y_test,batch_size=batch_size)

print(f"Test loss = {test_result[0]}")
print(f"Test mae = {test_result[1]}")

