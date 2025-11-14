# Load prediction 

import pandas as pd
import numpy as np

df = pd.read_csv('loan_pred.csv')
print(df.head())
print(df.duplicated().sum())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x = df.drop(columns=['Index','Defaulted?'],axis=1)
# print(x.head())
# print(x.shape)

y = df['Defaulted?']
# print(y.head())

import matplotlib.pyplot as plt
plt.scatter(x.iloc[:,0], x.iloc[:,1], color='red')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(68,activation='relu',input_dim=3))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
fit = model.fit(x_train_scaled,y_train,epochs=10,validation_data=(x_test_scaled,y_test))
print(fit)

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())

prediction = model.predict(x_test_scaled)

prob_prediction = np.where(prediction>0.5,1,0)
print(prob_prediction)

from sklearn.metrics import accuracy_score
print("Accuracy score: ",accuracy_score(y_test,prob_prediction))

plt.plot(fit.history['loss'], label='Training Loss')
plt.plot(fit.history['val_loss'], label='Validation Loss')
plt.show()

plt.plot(fit.history['accuracy'], label='Training Accuracy')
plt.plot(fit.history['val_accuracy'], label='Validation Accuracy')
plt.show() 

