import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('insurance.csv')

# Check the columns to ensure 'charges' exists as the target
print("Dataset columns:", data.columns)

# Set 'charges' as the target variable
X = data.drop('charges', axis=1)  # Features: All columns except 'charges'
y = data['charges']  # Target: 'charges' column

# One-hot encode categorical features: 'sex', 'smoker', and 'region'
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

# Handle missing values if any
X.fillna(X.mean(), inplace=True)  # Replace missing values with mean for simplicity

# Feature scaling: Use StandardScaler to scale continuous features
scaler = StandardScaler()
X[['age', 'bmi', 'children']] = scaler.fit_transform(X[['age', 'bmi', 'children']])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV to find the best parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model and scaler for future use
joblib.dump(best_model, 'xgb_model_optimized.pkl')  # Save the model with a more descriptive name
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler used for feature scaling
