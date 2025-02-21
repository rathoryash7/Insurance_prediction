import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""# New Section"""

from google.colab import files
uploaded = files.upload()

insurance_dataset = pd.read_csv('insurance.csv')

# first 5 rows of the dataframe
insurance_dataset.head()

# number of rows and columns
insurance_dataset.shape

# getting some informations about the dataset
insurance_dataset.info()

# checking for missing values
insurance_dataset.isnull().sum()

# statistical Measures of the dataset
insurance_dataset.describe()

# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

insurance_dataset['sex'].value_counts()

# bmi distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

insurance_dataset['children'].value_counts()

# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()

insurance_dataset['smoker'].value_counts()

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()

insurance_dataset['region'].value_counts()

# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

#Data Pre-Processing

#Encoding the categorical features
# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

#Model Evaluation

# prediction on training data
training_data_prediction =regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

# prediction on test data
test_data_prediction =regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

#Building a Predictive System

input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])

# Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

# Mean Squared Error (MSE)
mse = metrics.mean_squared_error(Y_test, test_data_prediction)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Model Evaluation

# prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared value for training data
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value (training data):', r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared value for test data
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value (test data):', r2_test)

# Accuracy Metrics for test data
# Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)

# Mean Squared Error (MSE)
mse = metrics.mean_squared_error(Y_test, test_data_prediction)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Model Evaluation

# prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared value for training data
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value (training data):', r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared value for test data
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value (test data):', r2_test)

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

#XGBOOST

import numpy as np
import pandas as pd
import train_model as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Define a patched XGBRegressor to include the __sklearn_tags__ method
class PatchedXGBRegressor(xgb.XGBRegressor):
    def __sklearn_tags__(self):
        return {"non_deterministic": True}

# Set pandas option for future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Loading the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Features and target variable
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Define the patched XGBoost regressor
xg_reg = PatchedXGBRegressor(objective='reg:squarederror', eval_metric='rmse')

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, scoring='r2', cv=3, verbose=1, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, Y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the model with the best parameters
best_xg_reg = PatchedXGBRegressor(objective='reg:squarederror', eval_metric='rmse', **best_params)
best_xg_reg.fit(X_train, Y_train)

# Make predictions
train_predictions = best_xg_reg.predict(X_train)
test_predictions = best_xg_reg.predict(X_test)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

#Gradient Boosting Machines (GBM):

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Loading the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Gradient Boosting Machine model
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gbm.fit(X_train, Y_train)

# Make predictions
train_predictions = gbm.predict(X_train)
test_predictions = gbm.predict(X_test)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

# Predicting insurance cost for new data
input_data = (31, 1, 25.74, 0, 1, 0)  # Example input data
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_scaled = scaler.transform(input_data_reshaped)

# Make prediction
prediction = gbm.predict(input_data_scaled)
print('The insurance cost is USD ', prediction[0])

#Prediction of cost from Xgboost

import numpy as np
import pandas as pd
import train_model as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Loading the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the XGBoost regressor
xgb_reg = xgb.XGBRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='r2', cv=3, verbose=1, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, Y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the model with the best parameters
best_xgb_reg = xgb.XGBRegressor(**best_params)
best_xgb_reg.fit(X_train, Y_train)

# Make predictions
train_predictions = best_xgb_reg.predict(X_train)
test_predictions = best_xgb_reg.predict(X_test)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

# Making a prediction for a new input
input_data = (31, 1, 25.74, 0, 1, 0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_reshaped = scaler.transform(input_data_reshaped)

prediction = best_xgb_reg.predict(input_data_reshaped)
print('The insurance cost is USD ', prediction[0])

#Neural Network

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Loading the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Neural Network model
nn = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
nn.fit(X_train, Y_train)

# Make predictions
train_predictions = nn.predict(X_train)
test_predictions = nn.predict(X_test)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

# Predicting insurance cost for new data
input_data = (31, 1, 25.74, 0, 1, 0)  # Example input data
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_scaled = scaler.transform(input_data_reshaped)

# Make prediction
prediction = nn.predict(input_data_scaled)
print('The insurance cost is USD ', prediction[0])

#random forest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Loading the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Random Forest model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_reg.fit(X_train, Y_train)

# Make predictions
train_predictions = rf_reg.predict(X_train)
test_predictions = rf_reg.predict(X_test)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

#GRP

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Loading the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Gaussian Process Regression model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)

# Train the model
gpr.fit(X_train, Y_train)

# Make predictions
train_predictions, train_std = gpr.predict(X_train, return_std=True)
test_predictions, test_std = gpr.predict(X_test, return_std=True)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

#from gussion processor regressor
# Predicting insurance cost for new data
input_data = (31, 1, 25.74, 0, 1, 0)  # Example input data
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_scaled = scaler.transform(input_data_reshaped)

# Make prediction
prediction, prediction_std = gpr.predict(input_data_scaled, return_std=True)
print('The insurance cost is USD ', prediction[0])

#For random forest algorithm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
insurance_dataset = pd.read_csv('insurance.csv')

# Data Pre-Processing
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Random Forest model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_reg.fit(X_train, Y_train)

# Make predictions
train_predictions = rf_reg.predict(X_train)
test_predictions = rf_reg.predict(X_test)

# Evaluate the model
r2_train = r2_score(Y_train, train_predictions)
r2_test = r2_score(Y_test, test_predictions)
mae = mean_absolute_error(Y_test, test_predictions)
mse = mean_squared_error(Y_test, test_predictions)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f'R squared value (training data): {r2_train}')
print(f'R squared value (test data): {r2_test}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Converting R squared values to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f'Accuracy of the model on training data: {r2_train_percentage:.2f}%')
print(f'Accuracy of the model on test data: {r2_test_percentage:.2f}%')

# Prediction for a new input
input_data = (31, 1, 25.74, 0, 1, 0)  # Example input data
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_scaled = scaler.transform(input_data_reshaped)

# Make prediction
prediction = rf_reg.predict(input_data_scaled)
print('The insurance cost is USD ', prediction[0])

# Additional evaluation (optional)
# For test predictions, calculate and display residuals
residuals = Y_test - test_predictions
print("Residuals of the test set: ", residuals.head())

