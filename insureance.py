import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('insurance.csv')

# Check the structure of the dataset
print("Dataset Columns:", df.columns)
print("Dataset Shape:", df.shape)

# Convert categorical columns to numeric using LabelEncoder
categorical_columns = ['sex', 'smoker', 'region']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate features (X) and target (y)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].values  # Features
y = df['charges'].values  # Target

# Normalize the target variable if necessary
# Uncomment if you want to scale `charges`
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Classifier (SVC) model
sv = SVC(kernel='linear')

try:
    sv.fit(X_train, y_train)
    print("Model trained successfully!")
except ValueError as e:
    print("Error in training the model:", e)

# Save the trained model to a file using pickle
with open('insurance_model.pkl', 'wb') as model_file:
    pickle.dump(sv, model_file)

print("Model saved successfully!")
