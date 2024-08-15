import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
import scipy.stats as stats
import matplotlib.pyplot as plt

def calculate_and_plot_mrmr_rankings(dataset_path, output_csv_path):

    # Load your dataset
    data = pd.read_csv(dataset_path)

    # List of columns to drop
    columns_to_drop = ["Tester Code", "Status", "Strike Type", "Gender", "Km/H", "Finish Time"]

    # Convert 'Elastic Exchange' column to numeric
    data['Elastic Exchange'] = data['Elastic Exchange'].str.rstrip('%').astype('float')

    # Select rows where the 'status' column is 'Full Data'
    filtered_data = data[data['Status'] == 'Full data']

    # Drop the specified columns
    filtered_data.drop(columns=columns_to_drop, inplace=True)

    # Separate your data into features (X) and the target variable (y)
    X = filtered_data.drop(columns=['Max Speed'])
    y = filtered_data['Max Speed']

    # Calculate mutual information between each feature and the target variable
    mi_scores = mutual_info_regression(X, y)

    # Create a DataFrame to store feature names and their MI scores
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})

    # Sort features by MI score in descending order
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

    # Calculate mutual information between each pair of features
    pairwise_mi_matrix = pd.DataFrame(index=X.columns, columns=X.columns)
    for feature1 in X.columns:
        for feature2 in X.columns:
            if feature1 != feature2:
                mi = normalized_mutual_info_score(X[feature1], X[feature2])
                pairwise_mi_matrix.at[feature1, feature2] = mi

    # Calculate MRMR values for each feature using the formula
    mrmr_values = []
    for feature in X.columns:
        mi_feature_target = mi_df.loc[mi_df['Feature'] == feature, 'MI_Score'].values[0]
        mi_feature_set = sum(mi_df.loc[mi_df['Feature'] != feature, 'MI_Score'])
        mrmr_value = mi_feature_target / (mi_feature_set - mi_feature_target)
        mrmr_values.append((feature, mrmr_value))

    # Create a DataFrame to store MRMR values for each feature
    mrmr_df = pd.DataFrame(mrmr_values, columns=['Feature', 'MRMR_Score'])

    # Sort features by MRMR score in descending order
    mrmr_df = mrmr_df.sort_values(by='MRMR_Score', ascending=False)

    # Plot MRMR values on a bar graph with features on the y-axis
    plt.figure(figsize=(10, 6))
    plt.barh(mrmr_df['Feature'], mrmr_df['MRMR_Score'])
    plt.xlabel('MRMR Score')
    plt.ylabel('Feature')
    plt.title('MRMR Scores for Features')
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=8)
    plt.show()

    # Create a list of tuples containing feature and rank
    feature_ranking_list = [(feature, rank + 1) for rank, feature in enumerate(mrmr_df['Feature'])]

    # Create a DataFrame to store the MRMR rankings
    mrmr_rankings_df = pd.DataFrame(feature_ranking_list, columns=['Feature', 'MRMR_Rank'])

    # Save MRMR rankings to a CSV file
    mrmr_rankings_df.to_csv(output_csv_path + '_Rankings.csv', index=False)

    # Print the list of feature rankings
    print("Feature Rankings based on MRMR Scores:")
    for feature, rank in feature_ranking_list:
        print(f"Feature: {feature}, Rank: {rank}")

    return feature_ranking_list

# Define the dataset path and output CSV path
dataset_path = 'RunnersDataV3.csv'
output_csv_path = 'MRMR'

calculate_and_plot_mrmr_rankings(dataset_path, output_csv_path)