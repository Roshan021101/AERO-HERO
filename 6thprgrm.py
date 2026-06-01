import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample Dataset
data = {
    'Name': ['Rahul', 'Priya', 'Amit', 'Sneha', 'Rohan'],
    'Income': [50000, 60000, None, 80000, 70000],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [25, None, 30, 28, 35]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df[['Income', 'Age']] = imputer.fit_transform(df[['Income', 'Age']])

print("\nDataset After Handling Missing Values:")
print(df)

# Encoding Categorical Variables
label_encoder = LabelEncoder()

df['Name'] = label_encoder.fit_transform(df['Name'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

print("\nDataset After Encoding:")
print(df)

# Feature Scaling
scaler = StandardScaler()
df[['Income', 'Age']] = scaler.fit_transform(df[['Income', 'Age']])

print("\nDataset After Feature Scaling:")
print(df)