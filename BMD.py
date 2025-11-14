import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, Back
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv('BMW_Car_Sales_Classification.csv')
print(df.head())
print(df.duplicated().sum())

# Find columns that contain strings
string_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).any()]
print(Fore.BLUE + Back.RED + "Columns containing strings:" + Style.RESET_ALL)
print(string_columns)

# Convert Sales_Classification to binary (0, 1)
df['Sales_Classification'] = df['Sales_Classification'].map({'Low': 0, 'High': 1})

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission'])
print(Fore.CYAN + Back.YELLOW + "In Numeric Form:" + Style.RESET_ALL)
print(df.head())

# Drop unwanted column
df = df.drop(columns='Sales_Volume', axis=1)
print(Fore.RED + Back.CYAN + "After dropping columns:" + Style.RESET_ALL)
print(df.head())

# Separate X and y
print("------------" + Fore.GREEN + Back.RED + "Dependent and Independent Variable" + Style.RESET_ALL + "---------------")
x = df.drop(columns='Sales_Classification', axis=1)
y = df['Sales_Classification']

print(Fore.RED + "Independent variables" + Style.RESET_ALL)
print(x.head())
print(x.shape)
print(Fore.RED + "Dependent variable" + Style.RESET_ALL)
print(y.head())

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Model
model = Sequential()

# ANN layers

model.add(Dense(128, input_shape=(33,), kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(Dense(62, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

print(Fore.GREEN + Back.BLACK + "Summary" + Style.RESET_ALL)
print(model.summary())

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train
model_fit = model.fit(
    x_train_scaled, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop],
    batch_size=32
)

# Predictions
predictions = model.predict(x_test_scaled)
predictions_binary = (predictions > 0.5).astype(int)

# Accuracy
print("Accuracy score:", accuracy_score(y_test, predictions_binary))
