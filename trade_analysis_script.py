import pandas as pd
import pytz
from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pandas.tseries.offsets import Minute
import seaborn as sns

def select_file():
    """
    Opens a file dialog to select a .txt file for analysis.
    :return: The filepath of the selected file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window.
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    root.destroy()  # Close the Tkinter instance properly.
    return file_path

def load_data(filepath, usecols):
    """
    Load the trade log data from a specified filepath and convert 'TransDateTime' from UTC to Eastern Time.
    :param filepath: Path to the trade log file.
    :param usecols: Columns of interest.
    :return: DataFrame with the loaded data and adjusted 'TransDateTime'.
    """
    df = pd.read_csv(filepath, sep='\t', usecols=usecols)
    
    # Convert 'TransDateTime' to datetime objects
    df['TransDateTime'] = pd.to_datetime(df['TransDateTime'], errors='coerce')
    
    # Localize the timezone to UTC since the original times are in UTC
    df['TransDateTime'] = df['TransDateTime'].dt.tz_localize('UTC')
    
    # Convert from UTC to Eastern Time, taking into account DST
    eastern = pytz.timezone('US/Eastern')
    df['TransDateTime'] = df['TransDateTime'].dt.tz_convert(eastern)

    return df

def preprocess_data(df):
    """
    Preprocess the data by adjusting prices and filtering out unwanted symbols.
    :param df: DataFrame with the loaded data.
    :return: Adjusted and filtered DataFrame.
    """
    price_columns = ['FillPrice', 'HighDuringPosition', 'LowDuringPosition']
    df[price_columns] = df[price_columns] / 100
    df = df[~df['Symbol'].str.contains('/NQ', na=False)]
    return df

def adjust_nat_times(df):
    """
    Adjusts NaT in 'TransDateTime' by adding 1 minute to the open time or subtracting 1 minute from the close time.
    This function assumes that each 'close' has a preceding 'open' and attempts to fix NaT values accordingly.
    :param df: DataFrame with the trade data.
    :return: DataFrame with adjusted 'TransDateTime' for NaT values.
    """
    open_indices = df[df['OpenClose'] == 'open'].index
    close_indices = df[df['OpenClose'] == 'close'].index

    # Loop through close indices to adjust NaT values based on the open time or the next close time
    for close_index in close_indices:
        if pd.isna(df.at[close_index, 'TransDateTime']):
            # Find the preceding open trade's index for the current close trade
            preceding_open_index = open_indices[open_indices < close_index].max()
            # If there's a valid open trade before this close trade, add 1 minute to its time
            if pd.notna(preceding_open_index):
                df.at[close_index, 'TransDateTime'] = df.at[preceding_open_index, 'TransDateTime'] + Minute(1)
            else:
                # If there's no valid open trade before, try to subtract 1 minute from the next valid close time
                following_close_index = close_indices[close_indices > close_index].min()
                if pd.notna(following_close_index):
                    df.at[close_index, 'TransDateTime'] = df.at[following_close_index, 'TransDateTime'] - Minute(1)

    return df

def calculate_pnl(df):
    """
    Calculates PnL for each trade, assuming 'TransDateTime' has been adjusted for NaT values.
    Adjusts 'TransDateTime' for NaT rows before calculation.
    :param df: DataFrame with trade data including 'TransDateTime', 'OpenClose', etc.
    :return: DataFrame with additional columns for PnL and cumulative PnL.
    """
    # First, adjust NaT values in 'TransDateTime'
    df = adjust_nat_times(df)
    df['PnL'] = 0.0  # Initialize PnL column
    df['ClosePrice'] = np.nan  # Initialize ClosePrice column
    open_trades = {}  # Dictionary to track open trades by symbol

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['OpenClose'].lower() == 'open':
            # Store open trade with its index to reference later
            if symbol not in open_trades:
                open_trades[symbol] = []
            open_trades[symbol].append((i, row))
        elif row['OpenClose'].lower() == 'close' and symbol in open_trades and open_trades[symbol]:
            # Pop the last open trade for this symbol
            open_index, open_row = open_trades[symbol].pop(0)
            # Calculate PnL using the open row and current close row
            pnl = calculate_trade_pnl(open_row, row)
            # Assign PnL and ClosePrice to the opening trade row for consistency in display
            df.at[open_index, 'PnL'] = pnl
            df.at[open_index, 'ClosePrice'] = row['FillPrice']  # Capture the closing FillPrice

    df['CumulativePnL'] = df['PnL'].cumsum()
    return df

def calculate_trade_pnl(open_row, close_row):
    """
    Calculate the profit or loss for a single trade.
    
    :param open_row: The row from the DataFrame representing the opening of the trade.
    :param close_row: The row from the DataFrame representing the closing of the trade.
    :return: The profit or loss for the trade.
    """
    tick_value = 1.25  # $ per tick
    ticks_per_point = 4  # ticks per point
    point_value = tick_value * ticks_per_point

    # Extract necessary information from the rows
    open_price = open_row['FillPrice']
    close_price = close_row['FillPrice']
    quantity = open_row['Quantity']
    
    # Determine the direction of the trade
    if open_row['BuySell'].lower() == 'buy':
        # Profit (or loss) calculation for a Buy trade
        pnl = (close_price - open_price) * quantity * point_value
    else:
        # Profit (or loss) calculation for a Sell trade
        pnl = (open_price - close_price) * quantity * point_value
    
    return pnl

def plot_pnl(df):
    # Convert timezone-aware datetime objects to naive for plotting
    df['TransDateTime'] = df['TransDateTime'].dt.tz_localize(None)

    plt.figure(figsize=(10, 6))
    plt.plot(df['TransDateTime'], df['CumulativePnL'], marker='o', linestyle='-')
    plt.title('Cumulative Profit and Loss Over Time')
    plt.xlabel('Transaction Time')
    plt.ylabel('Cumulative P&L ($)')

    num_bins = max(1, len(df) // 20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=num_bins))

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_pnl_table(df, num_trades=10):
    """
    Display a table of the profit/loss calculations for the first 'num_trades' trades.
    :param df: DataFrame with calculated P&L.
    :param num_trades: Number of trades to display.
    """
    # Filter the DataFrame to only include rows where PnL calculation was performed
    # and limit the number of rows based on 'num_trades'
    df_display = df[df['PnL'] != 0].head(num_trades)[
        ['TransDateTime', 'Symbol', 'BuySell', 'FillPrice', 'ClosePrice', 'Quantity', 'PnL', 'CumulativePnL']]
    print(df_display)

def find_nat_rows(df):
    """
    Identifies rows in the DataFrame where 'TransDateTime' is NaT.
    :param df: DataFrame with 'TransDateTime' column.
    :return: DataFrame with only the rows where 'TransDateTime' is NaT.
    """
    nat_rows = df[pd.isna(df['TransDateTime'])]
    return nat_rows

def aggregate_profit_by_time(df):
    # Extract hour and day name from 'TransDateTime'
    df['Hour'] = df['TransDateTime'].dt.hour
    df['DayOfWeek'] = df['TransDateTime'].dt.day_name()
    
    # Aggregate data to calculate total or average profit by day and hour
    profit_summary = df.groupby(['DayOfWeek', 'Hour'])['PnL'].sum().unstack(fill_value=0)
    
    # Sort the index based on day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    profit_summary = profit_summary.reindex(day_order)
    
    return profit_summary

def plot_heatmap(profit_summary):
    plt.figure(figsize=(12, 8))
    # Use the 'RdYlGn' colormap and center the color map at 0 to differentiate positive and negative values
    sns.heatmap(profit_summary, annot=True, cmap='RdYlGn', center=0, fmt=".1f")
    plt.title('Profitability Heatmap by Day of Week and Hour of Day')
    plt.ylabel('Day of Week')
    plt.xlabel('Hour of Day')
    plt.xticks(rotation=45)
    plt.show()

def plot_pnl_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PnL'], kde=True, color='blue')  # Histogram with KDE
    plt.title('Distribution of Trade Profit and Loss')
    plt.xlabel('Profit and Loss')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def create_pnl_analysis_table(df):
    # Define PnL ranges
    pnl_ranges = [(-40, -30), (-30, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 30), (30, 40)]
    # List to store dictionaries before creating DataFrame
    data = []

    for pnl_range in pnl_ranges:
        # Filter trades within the current PnL range
        filtered_trades = df[(df['PnL'] > pnl_range[0]) & (df['PnL'] <= pnl_range[1])]
        # Append the count to the data list
        data.append({'PnL Range': f'{pnl_range[0]} to {pnl_range[1]}', 
                     'Count of Trades': filtered_trades.shape[0]})

    # For trades losing or gaining more than the specified ranges
    more_loss = df[df['PnL'] <= -40].shape[0]
    more_gain = df[df['PnL'] > 40].shape[0]
    data.append({'PnL Range': '<= -40', 'Count of Trades': more_loss})
    data.append({'PnL Range': '> 40', 'Count of Trades': more_gain})
    
    # Create DataFrame from the list of dictionaries
    analysis_df = pd.DataFrame(data)

    return analysis_df

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

        # Assuming 'df' is your DataFrame after attempting to convert 'TransDateTime'
        nat_rows = find_nat_rows(df)

        if not nat_rows.empty:
            print("Rows with NaT in 'TransDateTime':")
            print(nat_rows)
        else:
            print("No rows with NaT in 'TransDateTime'.")

        # Preprocess the data
        df_preprocessed = preprocess_data(df)

        # Calculate P&L and add to DataFrame
        df_with_pnl = calculate_pnl(df_preprocessed)

        # Display the table of calculations for the first 10 trades
        display_pnl_table(df_with_pnl, 10)

        # Plot cumulative P&L
        plot_pnl(df_with_pnl)   

        profit_summary = aggregate_profit_by_time(df_with_pnl)
        plot_heatmap(profit_summary)
        plot_pnl_distribution(df_with_pnl)
        pnl_analysis_table = create_pnl_analysis_table(df_with_pnl)
        print(pnl_analysis_table)

    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
