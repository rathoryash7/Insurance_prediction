import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from ml.constants import MODEL_PATH, SCALER_PATH
from ml.preprocess import prepare_training_features

# Load data
data = pd.read_csv("insurance.csv")

X = data.drop("charges", axis=1)
y = data["charges"]

X, scaler = prepare_training_features(X, fit_scaler=True)
X.fillna(X.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 10],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb_model = xgb.XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_params_
print("Best hyperparameters:", best_model)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

joblib.dump(grid_search.best_estimator_, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
