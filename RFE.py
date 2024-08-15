import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('RunnersDataV3.csv')

# Data preprocessing steps
data['Elastic Exchange'] = data['Elastic Exchange'].str.rstrip('%').astype('float')
filtered_data = data[data['Status'] == 'Full data']
columns_to_drop = ["Tester Code", "Status", "Strike Type", "Gender", "Km/H", "Finish Time"]
filtered_data.drop(columns=columns_to_drop, inplace=True)

# Separate data into features (X) and target variable (y)
X = filtered_data.drop(columns=['Max Speed'])
y = filtered_data['Max Speed']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize Sequential Feature Selector
sfs = SFS(rf, 
          k_features=20, 
          forward=False, 
          floating=False, 
          verbose=2,
          scoring='neg_mean_squared_error',
          cv=5)

# Fit SFS to your training data
sfs = sfs.fit(X_train, y_train)

# Get the selected feature indices and print them
selected_feature_indices = list(sfs.k_feature_idx_)
print(f'Selected features indices: {selected_feature_indices}')

# Get the selected feature names
selected_features = X_train.columns[list(sfs.k_feature_idx_)]

# Train a random forest model on the selected features
X_train_sfs = sfs.transform(X_train)
rf.fit(X_train_sfs, y_train)

# Transform the test set to match the selected features and make predictions
X_test_sfs = sfs.transform(X_test)
y_pred = rf.predict(X_test_sfs)

# Calculate and print the MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature importances and ranking
feature_importances = rf.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), selected_features[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances of Selected Features')
plt.show()

# Create a DataFrame and rank features based on importance
feature_rank_df = pd.DataFrame({
    'Feature': selected_features[sorted_idx],
    'Importance': feature_importances[sorted_idx]
}).sort_values(by='Importance', ascending=False)

feature_rank_df['Rank'] = range(1, len(feature_rank_df) + 1)
feature_rank_df.to_csv('RF_Rankings.csv', index=False)
