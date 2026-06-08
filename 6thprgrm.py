# write a program to handle missing data encode categorial variablesand program featurescaling 

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.DataFrame({
    'Name': ['Rahul', 'Priya', 'Amit', 'Sneha', 'Rohan'],
    'Income': [50000, 60000, None, 80000, 70000],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [25, None, 30, 28, 35]
})

print("Original Dataset:\n", df)

df[['Income', 'Age']] = SimpleImputer(strategy='mean').fit_transform(df[['Income', 'Age']])
print("\nDataset After Handling Missing Values:\n", df)

for col in ['Name', 'Gender']:
    df[col] = LabelEncoder().fit_transform(df[col])
print("\nDataset After Encoding:\n", df)

df[['Income', 'Age']] = StandardScaler().fit_transform(df[['Income', 'Age']])
print("\nDataset After Feature Scaling:\n", df)
