import pandas as pd
import tkinter as tk
from tkinter import filedialog

def select_file():
    """
    Opens a file dialog to select a .txt file for analysis.
    :return: The filepath of the selected file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window.
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    return file_path

def load_data(filepath, usecols):
    """
    Load the trade log data from a specified filepath.
    :param filepath: Path to the trade log file.
    :param usecols: Columns of interest.
    :return: DataFrame with the loaded data.
    """
    return pd.read_csv(filepath, sep='\t', usecols=usecols)

def preprocess_data(df):
    """
    Preprocess the data by adjusting prices and filtering out unwanted symbols.
    :param df: DataFrame with the loaded data.
    :return: Adjusted and filtered DataFrame.
    """
    # Adjust prices
    df['FillPrice'] = df['FillPrice'] / 100
    df['HighDuringPosition'] = df['HighDuringPosition'] / 100
    df['LowDuringPosition'] = df['LowDuringPosition'] / 100

    # Filter out /NQ symbols
    df = df[~df['Symbol'].str.contains('/NQ', na=False)]

    return df

def add_analysis_features(df):
    """
    Placeholder for additional analysis features.
    :param df: Preprocessed DataFrame.
    :return: DataFrame with additional analysis features.
    """
    # Example: df['NewFeature'] = df['Column'].apply(lambda x: ...)
    return df

def main():
    usecols = [
        'TransDateTime',
        'Symbol',
        'Quantity',
        'BuySell',
        'FillPrice',
        'OpenClose',
        'HighDuringPosition',
        'LowDuringPosition'
    ]

    # Select the file for analysis
    filepath = select_file()

    if filepath:  # Proceed if a file was selected
        # Load the data
        df = load_data(filepath, usecols)

        # Preprocess the data
        df_preprocessed = preprocess_data(df)

        # Add any additional analysis features
        df_analyzed = add_analysis_features(df_preprocessed)

        # Display or save your results
        print(df_analyzed.head())
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()