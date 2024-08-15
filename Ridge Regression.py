import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load your dataset
data = pd.read_csv('RunnersDataV3.csv')

# Data preprocessing steps
columns_to_drop = ["Tester Code", "Status", "Strike Type", "Gender", "Km/H", "Finish Time", "Fore Foot", "Mid Foot", "Rear Foot", "Female", "Male"]
data['Elastic Exchange'] = data['Elastic Exchange'].str.rstrip('%').astype('float')
filtered_data = data[data['Status'] == 'Full data']
filtered_data.drop(columns=columns_to_drop, inplace=True)

# Separate data into features (X) and target variable (y)
X = filtered_data.drop(columns=['Max Speed'])
y = filtered_data['Max Speed']

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize and train the Ridge Regression model
ridge_regressor = Ridge(random_state=42, alpha=60)
ridge_regressor.fit(X_train, y_train)

# Perform Permutation Feature Importance
perm_importance = permutation_importance(ridge_regressor, X_test, y_test, n_repeats=30, random_state=42)

# Create a DataFrame for feature importances
feature_importances = perm_importance.importances_mean
feature_names = X.columns  # Using original feature names
feature_rank_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Add a rank column
feature_rank_df['ridge_Rank'] = range(1, len(feature_rank_df) + 1)

# Select only Feature and Rank for the CSV
feature_rank_df = feature_rank_df[['Feature', 'ridge_Rank']]

# Save to CSV
feature_rank_df.to_csv('Ridge_Rankings.csv', index=False)

# Optionally, display the DataFrame
print(feature_rank_df)

y_pred = ridge_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Regression)')
line_of_best_fit = np.linspace(min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred)), 100)
plt.plot(line_of_best_fit, line_of_best_fit, color='red', linestyle='--', linewidth=2, label='Line of Best Fit')

plt.legend()
plt.savefig('Ridge Plot.png', bbox_inches='tight')
plt.show()

ridge_regressor = Ridge()

# Define the grid of hyperparameters to search
param_grid = {'alpha': [0.1, 1, 10, 20, 50, 100]}

# Create the GridSearchCV object
grid_search = GridSearchCV(ridge_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Get the best Ridge Regression model
best_ridge_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred_grid_search = best_ridge_model.predict(X_test)

# Evaluate the model's performance using metrics
mse_grid_search = mean_squared_error(y_test, y_pred_grid_search)
rmse_grid_search = mse_grid_search**0.5
mae_grid_search = mean_absolute_error(y_test, y_pred_grid_search)

print(f'Mean Squared Error (MSE) with Grid Search: {mse_grid_search}')
print(f'Root Mean Squared Error (RMSE) with Grid Search: {rmse_grid_search}')
print(f'Mean Absolute Error (MAE) with Grid Search: {mae_grid_search}')

# Make predictions on the test set using the best Ridge Regression model
y_pred_grid_search = best_ridge_model.predict(X_test)

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred_grid_search, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Regression) - Ridge Regression with Grid Search')

# Plot the line of best fit
line_of_best_fit = np.linspace(min(min(y_test), min(y_pred_grid_search)), max(max(y_test), max(y_pred_grid_search)), 100)
plt.plot(line_of_best_fit, line_of_best_fit, color='red', linestyle='--', linewidth=2, label='Line of Best Fit')

plt.legend()
plt.savefig('Ridge Grid Search Plot.png', bbox_inches='tight')
plt.show()

feature_names = X.columns  # Assuming X is a DataFrame

# Create a DataFrame for feature importances
feature_importances = perm_importance.importances_mean
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the bar plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Permutation Feature Importance - Ridge Regression')
plt.savefig('Ridge Feature Importance.png', bbox_inches='tight')
plt.show()

# Define labels and values for MSE, RMSE, and MAE
labels = ['MSE', 'RMSE', 'MAE']
ridge_metrics = [mse, rmse, mae]
grid_search_metrics = [mse_grid_search, rmse_grid_search, mae_grid_search]

# Plot the bar plot
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, ridge_metrics, width, label='Ridge Regression')

# Add some text for labels, title and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of the bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.savefig('Ridge Evaluation Metrics.png', bbox_inches='tight')
plt.show()