# Regression Proble using Dl 
'''
target --> maximum temperature of the day

unncessary columns --> STA ,YR, MO, DA
 
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Read CSV
df = pd.read_csv('Summary of Weather.csv')

# Drop unnecessary columns
df.drop(['STA', 'YR', 'MO', 'DA'], axis=1, inplace=True)

# Replace non-numeric placeholders
df.replace('T', 0, inplace=True)         # 'Trace' → 0
df.replace('#VALUE!', 1, inplace=True)   # Error marker → 1

# Convert all columns to string temporarily to handle mixed types
df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

# Fix '1 1' type values → '1.1'
df = df.replace(r'^(\d+)\s+(\d+)$', r'\1.\2', regex=True)

# Convert all to numeric (non-convertible → NaN)
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Label encode Date
label = LabelEncoder()
df['Date_encoded'] = label.fit_transform(df['Date'].astype(str))

# Drop original Date
df.drop(columns=['Date'], inplace=True)

# Fill any NaN values with 0
df.fillna(0, inplace=True)

# Features & target
x = df.drop(columns='MaxTemp', axis=1)
y = df['MaxTemp']


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Before scaling:\n", x_test)
print("After scaling:\n", x_test_scaled)


import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(7, activation='relu', input_dim=26))
model.add(Dense(7, activation='relu'))
model.add(Dense(1,activation='linear'))  # For regression problem we use linear activation

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) # mean absolute error
history  = model.fit(x_train_scaled, y_train, epochs=10, batch_size=10, validation_split=0.2)

print(history)

prediction = model.predict(x_test_scaled)
print(prediction)

from sklearn.metrics import r2_score

print("r2Score: ",r2_score(y_test,prediction))