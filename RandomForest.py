import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from DataPreprocessing1 import load_and_prepare_data

# Specify the file paths
rankings_file = "Final_Rankings.csv"
runner_data_file = "RunnersDataV3.csv"

# Load and prepare data
X, y = load_and_prepare_data(rankings_file, runner_data_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest Regressor with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 500], # Number of trees in the forest
    'max_depth': [2, 3, 4, 5], # Maximum depth of the trees
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4] # Minimum number of samples required to be at a leaf node
}
rf_regressor = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Extract the best model and its hyperparameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Output the evaluation metrics
print("Best Hyperparameters:", best_params)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Plotting the metrics for visual representation
metrics = ['MSE', 'MAE', 'RMSE']
values = [mse, mae, rmse]
plt.bar(metrics, values)
plt.title('Model Evaluation Metrics for Random Forest Regressor')
plt.show()
