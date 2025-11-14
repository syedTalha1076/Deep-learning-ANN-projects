

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')
# print(df.head())
# print(df.shape)
# print(df.duplicated().sum())
dropDf = df.drop_duplicates()
# print(dropDf.duplicated().sum())

# print(dropDf.keys())
x = dropDf.drop(columns=['target'])
print(x.shape)
y = dropDf['target']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(
    [
        Dense(60,activation = 'relu',input_shape =(13,)),
        Dense(32,activation = 'relu'),
        Dense(1,activation = 'sigmoid')
    ]
)


model.add(Dense(11,activation = 'relu',input_dim = 13))
model.add(Dense(11,activation = 'relu'))
model.add(Dense(1,activation = 'relu'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # loss ='binary_crossentropy' is used for binary classification problems, 'adam' is an optimizer that adjusts the learning rate during training, and 'accuracy' is a metric to evaluate the model's performance.
fit = model.fit(x_train_scaled, y_train, epochs=100, batch_size=50, validation_data=(x_test_scaled, y_test))  # train the model for 100 epochs with batch size of 32 and validation data
print(fit)

print(model.layers[0].get_weights())  # get weights of the first layer
print(model.layers[1].get_weights())  # get weights of the second layer
print(model.layers[2].get_weights())  # get weights of the third layer

prediction = model.predict(x_test_scaled)  # make predictions on the test data 
# print("Prediction:\n", prediction[:10])  

prediction = np.where(prediction > 0.5,1,0)
print(prediction)

from sklearn.metrics import accuracy_score
print("accuracy score:\n", accuracy_score(y_test, prediction))  # calculate accuracy score

# import matplotlib.pyplot as plt
# plt.plot(fit.history['loss'], label='Training Loss')
# plt.plot(fit.history['val_loss'], label='Validation Loss')
# plt.show()

# plt.plot(fit.history['accuracy'], label='Training Accuracy')
# plt.plot(fit.history['val_accuracy'], label='Validation Accuracy')
# plt.show() 