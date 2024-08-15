import pandas as pd

def load_and_prepare_data(rankings_file, runner_data_file):
    # Load and sort rankings
    df_rankings = pd.read_csv(rankings_file).sort_values(by="Average_Rank").head(20)
    
    # Load runner data and preprocess
    df_runner_data = pd.read_csv(runner_data_file)
    df_runner_data['Elastic Exchange'] = pd.to_numeric(df_runner_data['Elastic Exchange'].str.rstrip('%'), errors='coerce')
    filtered_data = df_runner_data[df_runner_data['Status'] == 'Full data']
    
    # Select features and prepare dataset
    selected_feature_names = pd.concat([df_rankings["Feature"], pd.Series(["Max Speed"])])
    df_selected_data = filtered_data[selected_feature_names]
    X = df_selected_data.drop("Max Speed", axis=1)
    y = df_selected_data["Max Speed"]
    
    return X, y
