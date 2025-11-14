# Multiclass classification 

import pandas as pd
import numpy as np

df = pd.read_csv('zoo2.csv')
print(df.head())

for col in df.columns:
    print(col)

string_cols_check = df.select_dtypes(include=['object']).columns
print(list(string_cols_check))

if(len(string_cols_check) > 0):
    print("Yes Present String columns")

else:
    print("Not present string columns")

print(df.duplicated().sum())


print(df.head())
# print(df['animal_name'].head())

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_animal = vectorizer.fit_transform(df["Animal_Names"])

# Convert to DataFrame
df_animal = pd.DataFrame(X_animal.toarray(), columns=vectorizer.get_feature_names_out())

# Concatenate with original dataset (without dropping Animal_Names)
df_final = pd.concat([df, df_animal], axis=1)

print("Final DataFrame Columns:", df_final.columns)
print(df_final.head())


corr = df.corr()

# Take only target correlation (drop itself)
# target_corr = corr[''].drop('concrete_compressive_strength')

# # Convert correlation to percentage effect (absolute value)
# effect_percentage = (target_corr.abs() * 100).sort_values(ascending=False)
# print(effect_percentage)