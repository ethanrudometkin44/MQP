import pandas as pd

# Read the data
df_runner_data = pd.read_csv("RunnersDataV3.csv")
df_description = pd.read_csv("List of Data Fields.csv")

# Since you want to filter by 'Status', do it before dropping the 'Status' column
filtered_data = df_runner_data[df_runner_data['Status'] == 'Full data']

# Now that you've filtered, drop the specified columns, including 'Status'
columns_to_drop = ['Tester Code', 'Status', 'Strike Type', 'Fore Foot', 'Mid Foot', 'Rear Foot', 'Finish Time', 'Gender', 'Male', 'Female']
filtered_data = filtered_data.drop(columns=columns_to_drop)

# Convert 'Elastic Exchange' column to numeric on the filtered data
filtered_data['Elastic Exchange'] = filtered_data['Elastic Exchange'].str.rstrip('%').astype('float')

# Calculate statistics
statistics = filtered_data.describe().loc[['min', 'max', 'mean', 'std']]

# Transpose the DataFrame for better readability
statistics = statistics.T

# Reset index to make 'Features' a column
statistics.reset_index(inplace=True)

# Rename the index column to 'Features'
statistics = statistics.rename(columns={'index': 'Feature'})

# Merge the statistics and df_description dataframes on the 'Features' column
result_df = statistics.merge(df_description, on='Feature')

# Convert the merged DataFrame to CSV
result_df.to_csv('SummaryStatistics.csv', index=False)

