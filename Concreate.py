# Regression type problem

import pandas as pd
import numpy as np

df = pd.read_csv('concrete_data.csv')
print(df.head())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.head())
print(df.duplicated().sum())

for col in df.columns:
    print(col)

# Check string columns 
string_cols_check = df.select_dtypes(include=['object']).columns
print(list(string_cols_check))

if(len(string_cols_check > 0)):
    print("Present string columns")
else:
    print("Not present string columns")

# Check effect of cols
corr = df.corr()

# Take only target correlation (drop itself)
target_corr = corr['concrete_compressive_strength'].drop('concrete_compressive_strength')

# Convert correlation to percentage effect (absolute value)
effect_percentage = (target_corr.abs() * 100).sort_values(ascending=False)
print(effect_percentage)

# Depedent and independent columns
x = df.drop(columns=['concrete_compressive_strength'],axis=1)
print(x.head())
print(x.shape)

y = df['concrete_compressive_strength']
print(y.head())

import matplotlib.pyplot as plt

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='green')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

model.add(Dense(128,activation='relu',input_dim=8))

model.add(Dense(62,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(1,activation='linear'))

print(model.summary())

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
fit = model.fit(x_train_scaled,y_train,epochs=200,validation_data=(x_test_scaled,y_test))
print(fit)

y_pred = model.predict(x_test_scaled)
from sklearn.metrics import r2_score
print("R2 score: ",r2_score(y_test,y_pred)*100)

plt.plot(fit.history['mae'],label='Training mae')
plt.plot(fit.history['val_mae'],label='validation mae')
plt.show()
plt.plot(fit.history['loss'],label='Training loss')
plt.plot(fit.history['val_loss'],label='validation loss')
plt.show()

# ------------------- User Input Prediction -------------------
print("\nEnter values for prediction:")
cement = float(input("Cement: "))
blast_furnace_slag = float(input("Blast Furnace Slag: "))
fly_ash = float(input("Fly Ash: "))
water = float(input("Water: "))
superplasticizer = float(input("Superplasticizer: "))
coarse_aggregate = float(input("Coarse Aggregate: "))
fine_aggregate = float(input("Fine Aggregate: "))
age = float(input("Age (days): "))

user_input = np.array([[cement, blast_furnace_slag, fly_ash, water,
                        superplasticizer, coarse_aggregate, fine_aggregate, age]])

user_input_scaled = scaler.transform(user_input)

prediction = model.predict(user_input_scaled)
print(f"Predicted Concrete Compressive Strength: {prediction[0][0]:.2f} MPa")
