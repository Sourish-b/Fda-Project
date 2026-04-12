from glob import glob
import os

import pandas as pd


def load_integrated(filepath):
    """Load a single integrated CSV file and print quick diagnostics."""
    df = pd.read_csv(filepath)
    print("Integrated dataset shape:", df.shape)
    print("Integrated dataset dtypes:")
    print(df.dtypes)
    print("Integrated dataset head(3):")
    print(df.head(3))
    print("Integrated dataset null counts:")
    print(df.isnull().sum())
    return df


def load_all_states(folder_path):
    """Load all state files from a folder and combine into one DataFrame."""
    pattern = os.path.join(folder_path, "*.csv")
    file_paths = sorted(glob(pattern))

    # Statewise files in this dataset are stored as Excel files.
    if not file_paths:
        xlsx_pattern = os.path.join(folder_path, "*.xlsx")
        file_paths = sorted(glob(xlsx_pattern))

    if not file_paths:
        raise FileNotFoundError(f"No CSV or XLSX files found in: {folder_path}")

    state_dfs = []
    for file_path in file_paths:
        if file_path.lower().endswith(".csv"):
            state_df = pd.read_csv(file_path)
        else:
            state_df = pd.read_excel(file_path)
        if state_df.shape[0] == 0:
            print(f"Warning: 0 rows found in file: {os.path.basename(file_path)}")
        state_dfs.append(state_df)

    combined_df = pd.concat(state_dfs, ignore_index=True)
    print("Total statewise combined shape:", combined_df.shape)
    return combined_df


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    integrated_path = os.path.join(project_root, "Dataset", "renewable_energy_dataset.csv")
    states_folder = os.path.join(project_root, "Dataset", "STATEWISE_CLIMATE_RENEWABLEENERGY_DATA")

    print("Integrated file path:", integrated_path)
    print("State files folder path:", states_folder)

    integrated_df = load_integrated(integrated_path)
    states_df = load_all_states(states_folder)

    if integrated_df is not None and states_df is not None:
        print("Data loading completed successfully.")
