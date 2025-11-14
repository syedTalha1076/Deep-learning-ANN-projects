
# NOTE
#! There is problem in this code 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("mobileApp.csv")
print(df.head())
print(pd.unique(df['app_version']))

df = df.drop(columns=['review_id','review_text','review_language','num_helpful_votes','user_age','user_country','user_gender'],axis=1)
print(df.head())

# ? Check which cols contain string
# string_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x,str)).any()]
# print(string_cols)

df = pd.get_dummies(df,columns=['app_name','app_category','review_date','device_type','app_version'])
print(df.head())

# plt.scatter(df.iloc[:,0],df.iloc[:,1],color='red')
# plt.show()

x = df.iloc[:,:-1]
print("Indepedent value:\n",x)
print(x.shape)
y = df.iloc[:,-1]
print("Depedent value:\n",y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


import tensorflow 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping

import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    
    counter = 0

    for i in range(hp.Int('nums_layers',min_value=1,max_value=20)):

        if counter == 0:
            model.add(Dense(
                hp.Int('units'+str(i),min_value = 5,max_value=300,step = 5),
                activation = hp.Choice('activation'+str(i),values=['sigmoid','relu','tanh']),
                input_dim = 4661
            ))

            model.add(Dropout(hp.Choice('Dropouts'+str(i),values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
       
        else:
            model.add(Dense(
                hp.Int('units'+str(i),min_value = 5,max_value=300,step = 5),
                activation = hp.Choice('activation'+str(i),values=['sigmoid','relu','tanh']),
                input_dim = 4661
            ))

            model.add(Dropout(hp.Choice('Dropouts'+str(i),values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
       
        counter += 1

    model.add(Dense(1,activation = 'softmax'))

    model.compile(optimizer=hp.Choice('optimizer',values=['adam','rmsprop','sgd','nadam','adamdelta']),
                  loss ='categorical_crossentropy',
                  metrics=['accuracy'])
    

    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    overwrite=True,
    directory = 'myDir',
    project_name = 'MobileApp'
)

print(tuner.search(x_train_scaled,y_train,epochs=10,validation_data=(x_test_scaled,y_test)))

print(tuner.get_best_hyperparameters()[0].values)

model = tuner.get_best_models(num_models=1)[0]
print("Best model:\n",model)

early_stop = EarlyStopping(
    monitor='val_loss',   # What to monitor (can also be 'val_accuracy')
    patience=5,           # Number of epochs with no improvement before stopping
    restore_best_weights=True # Rollback to best model weights
)

fit = model.fit(x_train_scaled,y_train,epochs=100,initial_epoch=5,validation_data=(x_test_scaled,y_test),callbacks=early_stop)
print(fit)

prediction = model.predict(x_test_scaled)
