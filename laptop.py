import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("laptop.csv")
print(df.head())

string_cols = [cols for cols in df.columns if df[cols].apply(lambda x: isinstance(x,str)).any() ]
print(string_cols)

x = df.drop('Price',axis=1)
y = df['Price']
print(y.head())

x = x.drop(columns=['Unnamed: 0','OS'],axis=1)
print(x.head())
print("Shape of x: ")
print(x.shape)

x = pd.get_dummies(x,columns=['Model','Generation','Core','Ram','SSD','Display','Graphics','Warranty'])
y = pd.get_dummies(y,columns=['Price'])
print(y.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
print(x_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_test_scaled)

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten

import keras_tuner as kt


def build_model(hp):

    model = Sequential()

    counter = 0

    for i in range(hp.Int('nums_layers',min_value = 1, max_value = 10)):
        #! Input layer
        if counter == 0:
            model.add(Dense(
                hp.Int('units'+str(i),min_value = 8,max_value = 128 , step = 4),
                activation = hp.Choice('activation'+str(i),values = ['relu','sigmoid','tanh','linear']),
                input_shape = (1452,)
            ))

            model.add(Dropout(hp.Choice("Dropouts"+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        
        #! Output layer

        else:

            model.add(Dense(
                hp.Int('units'+str(i),min_value = 8,max_value = 128 , step = 4),
                activation = hp.Choice('activation'+str(i),values = ['relu','sigmoid','tanh','linear'])
            ))
            model.add(Dropout(hp.Choice("Dropouts"+str(i),values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))

        counter += 1

        model.add(Dense(1,activation = 'linear'))

        model.compile(optimizer=hp.Choice('optimizer',values=['adam','rmsprop','sgd','nadam','adadelta']),
                  loss ='mean_squared_error',
                  metrics=['mae'])
        
        return model
    
tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=3,
    overwrite=True,
    directory = 'laptopDir',
    project_name = 'Laptop Price Prediction'
)


tuner.search(x_train,y_train,epochs=10,validation_data=(x_test_scaled,y_test))

print("Best parameter:\n",tuner.get_best_hyperparameters()[0].values)

model = tuner.get_best_models(num_models=1)[0]

model_fit = model.fit(x_train_scaled,y_train,epochs=100,initial_epoch=6,validation_data=(x_test,y_test))

print("Model Training: ")
print(model_fit)

prediction = model.predict(x_test_scaled)

from sklearn.metrics import r2_score

# Ensure y_test is 1D
y_true = y_test.values.flatten() if hasattr(y_test, 'values') else y_test.flatten()

# Ensure prediction is 1D
y_pred = prediction.flatten()

# Compute R²
r2 = r2_score(y_true, y_pred)
print("R2 Score:", r2)




