import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from DataPreprocessing import load_and_prepare_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Specify the file paths
rankings_file = "Final_Rankings.csv"
runner_data_file = "RunnersDataV3.csv"

# Load and prepare data
X, y = load_and_prepare_data(rankings_file, runner_data_file)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and neural network
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('mlp', MLPRegressor(max_iter=1000))  # Feedforward Neural Network
])

# Define hyperparameters grid
param_grid = {
    'mlp__hidden_layer_sizes': [(15,10),(13,9),(10,5)],  # Various hidden layer configurations
    'mlp__activation': ['logistic', 'relu'],  # Activation functions
    'mlp__alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term)
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(n_splits=5))

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Get the best estimator
best_model = grid_search.best_estimator_

# Make predictions on test set
y_pred_test = best_model.predict(X_test)

# Calculate mean squared error on test set
mse_test = np.mean((y_pred_test - y_test) ** 2)
print("Mean Squared Error on Test Set:", mse_test)

# Calculate Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("Root Mean Squared Error on Test Set:", rmse_test)

# Calculate Mean Absolute Error (MAE)
mae_test = mean_absolute_error(y_test, y_pred_test)
print("Mean Absolute Error on Test Set:", mae_test)