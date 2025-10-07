from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from tkinter import filedialog, TclError
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pandas.api.types import DatetimeTZDtype
from pandas.tseries.offsets import Minute
import seaborn as sns


def _is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def format_currency(value):
    if _is_numeric(value) and np.isfinite(value):
        return f"${value:,.2f}"
    return "N/A"


def format_percentage(value):
    if _is_numeric(value) and np.isfinite(value):
        return f"{value * 100:.2f}%"
    return "N/A"


def format_ratio(value, decimals=2):
    if _is_numeric(value) and np.isfinite(value):
        return f"{value:.{decimals}f}"
    return "N/A"


def format_minutes(value, decimals=1):
    if _is_numeric(value) and np.isfinite(value):
        return f"{value:.{decimals}f} min"
    return "N/A"


def format_hours(value, decimals=2):
    if _is_numeric(value) and np.isfinite(value):
        return f"{value:.{decimals}f} hrs"
    return "N/A"

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
    trans_datetime = pd.to_datetime(df['TransDateTime'], errors='coerce')

    # Localize only if the timestamps are timezone-naïve
    if trans_datetime.dt.tz is None:
        trans_datetime = trans_datetime.dt.tz_localize('UTC')

    # Convert from UTC to Eastern Time, taking into account DST
    eastern = pytz.timezone('US/Eastern')
    df['TransDateTime'] = trans_datetime.dt.tz_convert(eastern)

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
    if isinstance(df['TransDateTime'].dtype, DatetimeTZDtype):
        df['CloseTime'] = pd.Series(pd.NaT, index=df.index, dtype=df['TransDateTime'].dtype)
    else:
        df['CloseTime'] = pd.NaT  # Track the closing timestamp per trade
    df['TradeDurationMinutes'] = np.nan  # Track how long each trade was open
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
            close_time = row['TransDateTime']
            if pd.notna(close_time):
                df.at[open_index, 'CloseTime'] = close_time
                if pd.notna(open_row['TransDateTime']):
                    trade_duration = (close_time - open_row['TransDateTime']).total_seconds() / 60
                    df.at[open_index, 'TradeDurationMinutes'] = trade_duration

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

def _save_figure(fig, output_dir, filename):
    image_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / filename
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
    return image_path


def plot_pnl(df, output_dir=None, show=True, figsize=(14, 8)):
    times = df['TransDateTime'].dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, df['CumulativePnL'], marker='o', linestyle='-')
    ax.set_title('Cumulative Profit and Loss Over Time')
    ax.set_xlabel('Transaction Time')
    ax.set_ylabel('Cumulative P&L ($)')

    num_bins = max(1, len(df) // 20)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=num_bins))

    fig.autofmt_xdate()
    ax.grid(True)
    fig.tight_layout()

    image_path = _save_figure(fig, output_dir, 'cumulative_pnl.png')

    if show:
        plt.show()
    plt.close(fig)

    return image_path

def display_pnl_table(df, num_trades=10, print_output=False, return_html=False):
    """
    Prepare a table of the profit/loss calculations for the first 'num_trades' trades.

    :param df: DataFrame with calculated P&L.
    :param num_trades: Number of trades to include.
    :param print_output: Whether to print the formatted table to stdout.
    :param return_html: Whether to also return the table rendered as HTML.
    :return: Tuple containing the formatted DataFrame and, optionally, the HTML string.
    """
    df_display = df[df['PnL'] != 0].head(num_trades)[
        [
            'TransDateTime',
            'CloseTime',
            'Symbol',
            'BuySell',
            'FillPrice',
            'ClosePrice',
            'Quantity',
            'PnL',
            'CumulativePnL',
            'TradeDurationMinutes'
        ]
    ].copy()

    formatted_df = df_display.copy()

    if not formatted_df.empty:
        formatted_df['TransDateTime'] = pd.to_datetime(formatted_df['TransDateTime']).dt.tz_localize(None)
        formatted_df['CloseTime'] = pd.to_datetime(formatted_df['CloseTime']).dt.tz_localize(None)

        formatted_df['TransDateTime'] = formatted_df['TransDateTime'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else 'N/A'
        )
        formatted_df['CloseTime'] = formatted_df['CloseTime'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else 'N/A'
        )

        formatted_df['FillPrice'] = formatted_df['FillPrice'].apply(format_currency)
        formatted_df['ClosePrice'] = formatted_df['ClosePrice'].apply(format_currency)
        formatted_df['PnL'] = formatted_df['PnL'].apply(format_currency)
        formatted_df['CumulativePnL'] = formatted_df['CumulativePnL'].apply(format_currency)
        formatted_df['TradeDurationMinutes'] = formatted_df['TradeDurationMinutes'].apply(
            lambda x: format_minutes(x, decimals=2) if pd.notna(x) else 'N/A'
        )

    if print_output:
        print(formatted_df)

    table_html = None
    if return_html:
        table_html = formatted_df.to_html(index=False, classes='styled-table', border=0, escape=False)

    return formatted_df, table_html

def find_nat_rows(df):
    """
    Identifies rows in the DataFrame where 'TransDateTime' is NaT.
    :param df: DataFrame with 'TransDateTime' column.
    :return: DataFrame with only the rows where 'TransDateTime' is NaT.
    """
    nat_rows = df[pd.isna(df['TransDateTime'])]
    return nat_rows

def aggregate_profit_by_time(df):
    df = df[df['PnL'] != 0].copy()

    # Use the close timestamp when available so realized PnL is attributed to
    # the time the trade was exited.  Fall back to the open timestamp for
    # defensive coverage (e.g., missing close times).
    if 'CloseTime' in df.columns:
        df['EffectiveTime'] = df['CloseTime'].where(df['CloseTime'].notna(), df['TransDateTime'])
    else:
        df['EffectiveTime'] = df['TransDateTime']

    # Extract hour and day name from the effective timestamp
    df['Hour'] = df['EffectiveTime'].dt.hour
    df['DayOfWeek'] = df['EffectiveTime'].dt.day_name()

    # Aggregate data to calculate total or average profit by day and hour
    profit_summary = df.groupby(['DayOfWeek', 'Hour'])['PnL'].sum().unstack(fill_value=0)

    # Sort the index based on day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    profit_summary = profit_summary.reindex(day_order)

    return profit_summary


def build_profitability_tables(profit_summary, df_with_pnl):
    filtered_df = df_with_pnl[df_with_pnl['PnL'] != 0].copy()

    profitability_html = profit_summary.to_html(classes='styled-table', border=0, float_format=lambda x: f"{x:,.2f}")

    daily_net_pnl = (
        filtered_df.groupby(filtered_df['TransDateTime'].dt.date)['PnL']
        .sum()
        .reset_index(name='Net PnL')
    )
    daily_net_pnl.columns = ['Date', 'Net PnL']
    daily_net_pnl['Net PnL'] = daily_net_pnl['Net PnL'].apply(format_currency)
    daily_net_pnl_html = daily_net_pnl.to_html(index=False, classes='styled-table', border=0)

    return profitability_html, daily_net_pnl_html

def plot_heatmap(profit_summary, output_dir=None, show=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(profit_summary, annot=True, cmap='RdYlGn', center=0, fmt=".1f", ax=ax)
    ax.set_title('Profitability Heatmap by Day of Week and Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_xlabel('Hour of Day')
    plt.setp(ax.get_xticklabels(), rotation=45)

    image_path = _save_figure(fig, output_dir, 'profitability_heatmap.png')

    if show:
        plt.show()
    plt.close(fig)

    return image_path

def plot_pnl_distribution(df, output_dir=None, show=True, figsize=(14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df['PnL'], kde=True, color='blue', ax=ax)
    ax.set_title('Distribution of Trade Profit and Loss')
    ax.set_xlabel('Profit and Loss')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    image_path = _save_figure(fig, output_dir, 'pnl_distribution.png')

    if show:
        plt.show()
    plt.close(fig)

    return image_path

def create_pnl_analysis_table(df):
    closed_trades = df[df['PnL'] != 0]
    # Define PnL ranges
    pnl_ranges = [(-40, -30), (-30, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 30), (30, 40)]
    # List to store dictionaries before creating DataFrame
    data = []

    for pnl_range in pnl_ranges:
        # Filter trades within the current PnL range
        filtered_trades = closed_trades[(closed_trades['PnL'] > pnl_range[0]) & (closed_trades['PnL'] <= pnl_range[1])]
        # Append the count to the data list
        data.append({'PnL Range': f'{pnl_range[0]} to {pnl_range[1]}', 
                     'Count of Trades': filtered_trades.shape[0]})

    # For trades losing or gaining more than the specified ranges
    more_loss = closed_trades[closed_trades['PnL'] <= -40].shape[0]
    more_gain = closed_trades[closed_trades['PnL'] > 40].shape[0]
    data.append({'PnL Range': '<= -40', 'Count of Trades': more_loss})
    data.append({'PnL Range': '> 40', 'Count of Trades': more_gain})
    
    # Create DataFrame from the list of dictionaries
    analysis_df = pd.DataFrame(data)

    return analysis_df


def generate_summary_statistics(df):
    closed_trades = df[df['PnL'] != 0]
    total_trades = int(closed_trades.shape[0])
    total_pnl = float(closed_trades['PnL'].sum()) if not closed_trades.empty else 0.0
    best_trade = float(closed_trades['PnL'].max()) if not closed_trades.empty else 0.0
    worst_trade = float(closed_trades['PnL'].min()) if not closed_trades.empty else 0.0
    if not closed_trades.empty:
        trading_days = closed_trades['TransDateTime'].dt.normalize().nunique()
        average_daily_pnl = total_pnl / trading_days if trading_days else 0.0
    else:
        trading_days = 0
        average_daily_pnl = 0.0

    return {
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'trading_days': trading_days,
        'average_daily_pnl': average_daily_pnl
    }


def calculate_performance_metrics(df):
    closed_trades = df[df['PnL'] != 0].copy()
    if closed_trades.empty:
        return {
            'expectancy': 0.0,
            'pnl_std': 0.0,
            'sharpe_ratio': float('nan'),
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': float('nan'),
            'reward_risk_ratio': float('nan'),
            'max_drawdown': 0.0,
            'average_trade_duration_minutes': float('nan'),
            'median_trade_duration_minutes': float('nan'),
            'total_exposure_minutes': 0.0,
            'exposure_hours': 0.0
        }

    closed_trades = closed_trades.sort_values('TransDateTime')
    pnl = closed_trades['PnL']
    total_trades = len(closed_trades)

    expectancy = pnl.mean()
    pnl_std = pnl.std(ddof=1) if total_trades > 1 else 0.0
    sharpe_ratio = (expectancy / pnl_std) * np.sqrt(total_trades) if pnl_std else float('nan')

    winning_trades = pnl[pnl > 0]
    losing_trades = pnl[pnl < 0]

    win_rate = len(winning_trades) / total_trades
    loss_rate = len(losing_trades) / total_trades
    average_win = winning_trades.mean() if not winning_trades.empty else 0.0
    average_loss = losing_trades.mean() if not losing_trades.empty else 0.0

    total_win = winning_trades.sum()
    total_loss = losing_trades.sum()
    profit_factor = total_win / abs(total_loss) if total_loss != 0 else float('inf')
    reward_risk_ratio = average_win / abs(average_loss) if average_loss != 0 else float('inf')

    cumulative_pnl = pnl.cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    closed_trades['CloseTime'] = pd.to_datetime(closed_trades['CloseTime'])
    trade_durations = (closed_trades['CloseTime'] - pd.to_datetime(closed_trades['TransDateTime'])).dropna()
    trade_duration_minutes = trade_durations.dt.total_seconds() / 60 if not trade_durations.empty else pd.Series(dtype=float)
    average_trade_duration = float(trade_duration_minutes.mean()) if not trade_duration_minutes.empty else float('nan')
    median_trade_duration = float(trade_duration_minutes.median()) if not trade_duration_minutes.empty else float('nan')
    total_exposure_minutes = float(trade_duration_minutes.sum()) if not trade_duration_minutes.empty else 0.0

    return {
        'expectancy': float(expectancy),
        'pnl_std': float(pnl_std),
        'sharpe_ratio': float(sharpe_ratio) if np.isfinite(sharpe_ratio) else float('nan'),
        'win_rate': float(win_rate),
        'loss_rate': float(loss_rate),
        'average_win': float(average_win) if np.isfinite(average_win) else 0.0,
        'average_loss': float(average_loss) if np.isfinite(average_loss) else 0.0,
        'profit_factor': float(profit_factor),
        'reward_risk_ratio': float(reward_risk_ratio),
        'max_drawdown': max_drawdown,
        'average_trade_duration_minutes': average_trade_duration,
        'median_trade_duration_minutes': median_trade_duration,
        'total_exposure_minutes': total_exposure_minutes,
        'exposure_hours': total_exposure_minutes / 60 if total_exposure_minutes else 0.0
    }


def generate_report(
    report_dir,
    summary_stats,
    performance_metrics,
    graph_paths,
    pnl_analysis_table,
    day_hour_summary_html,
    daily_net_pnl_html,
    trade_highlights_html=None
):
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'trade_analysis_report.html'

    graph_cards = []
    for title, path in graph_paths:
        if path is not None:
            graph_cards.append(
                (
                    "<article class='card graph-card'>"
                    f"<div class='graph-head'><h3>{title}</h3>"
                    f"<a href='{path.name}' target='_blank' class='graph-link'>View full size</a></div>"
                    f"<div class='graph-canvas'><img src='{path.name}' alt='{title}' loading='lazy'></div>"
                    "</article>"
                )
            )

    summary_swatches = [
        ("#38bdf8", "#0284c7"),
        ("#f97316", "#ea580c"),
        ("#22c55e", "#16a34a"),
        ("#a855f7", "#7c3aed"),
        ("#ec4899", "#db2777"),
        ("#facc15", "#eab308"),
    ]
    performance_swatches = [
        ("#6366f1", "#4338ca"),
        ("#0ea5e9", "#0284c7"),
        ("#f472b6", "#db2777"),
        ("#22d3ee", "#06b6d4"),
        ("#2dd4bf", "#14b8a6"),
        ("#f97316", "#ea580c"),
        ("#38bdf8", "#0ea5e9"),
        ("#facc15", "#eab308"),
        ("#c084fc", "#a855f7"),
        ("#34d399", "#059669"),
        ("#fb7185", "#f43f5e"),
        ("#f59e0b", "#d97706"),
        ("#e879f9", "#c026d3"),
    ]

    summary_items = [
        ('Total Closed Trades', summary_stats['total_trades'], '▣'),
        ('Trading Days Covered', summary_stats['trading_days'], '⏱'),
        ('Total Profit/Loss', format_currency(summary_stats['total_pnl']), '↑'),
        ('Average Daily P&L', format_currency(summary_stats['average_daily_pnl']), 'µ'),
        ('Best Trade', format_currency(summary_stats['best_trade']), '★'),
        ('Worst Trade', format_currency(summary_stats['worst_trade']), '▼'),
    ]

    def _render_stat_cards(items, card_class, swatches):
        card_markup = []
        for index, item in enumerate(items):
            label, value, icon, *custom_swatch = item
            swatch = custom_swatch[0] if custom_swatch else swatches[index % len(swatches)]
            start_color, end_color = swatch
            card_markup.append(
                "<div class='stat-card {card_class}' style='--swatch-start: {start}; --swatch-end: {end};'>".format(
                    card_class=card_class,
                    start=start_color,
                    end=end_color,
                )
                + f"<div class='stat-emblem'><span>{icon}</span></div>"
                + f"<div class='stat-content'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>"
                + "</div>"
            )
        return ''.join(card_markup)

    summary_html = _render_stat_cards(summary_items, 'summary-card', summary_swatches)

    performance_items = [
        ('Expectancy (per trade)', format_currency(performance_metrics['expectancy']), 'Σ'),
        ('Standard Deviation of P&L', format_currency(performance_metrics['pnl_std']), 'σ'),
        ('Sharpe Ratio (per trade)', format_ratio(performance_metrics['sharpe_ratio']), '∫'),
        ('Win Rate', format_percentage(performance_metrics['win_rate']), '✓'),
        ('Loss Rate', format_percentage(performance_metrics['loss_rate']), '✗'),
        ('Average Win', format_currency(performance_metrics['average_win']), '↗'),
        ('Average Loss', format_currency(performance_metrics['average_loss']), '↘'),
        ('Profit Factor', format_ratio(performance_metrics['profit_factor']), 'π'),
        ('Reward-to-Risk Ratio', format_ratio(performance_metrics['reward_risk_ratio']), '⁐'),
        ('Max Drawdown', format_currency(performance_metrics['max_drawdown']), '≤'),
        ('Average Trade Duration', format_minutes(performance_metrics['average_trade_duration_minutes']), '⏲'),
        ('Median Trade Duration', format_minutes(performance_metrics['median_trade_duration_minutes']), '⏳'),
        ('Total Market Exposure', format_hours(performance_metrics['exposure_hours']), '⦿'),
    ]
    performance_html = _render_stat_cards(performance_items, 'performance-card', performance_swatches)

    pnl_table_html = pnl_analysis_table.to_html(index=False, classes='styled-table', border=0)

    trade_highlights_section = ""
    if trade_highlights_html:
        trade_highlights_section = f"""
            <section class='highlights-section'>
                <div class='section-heading'>
                    <h2>Trade-Level Highlights</h2>
                    <p>Representative trade blotter excerpt spotlighting inflection points within the review period.</p>
                </div>
                <div class='card table-card'>
                    <div class='table-scroll'>
                        {trade_highlights_html}
                    </div>
                </div>
            </section>
        """


    html_content = f"""<!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
        <title>Trade Analysis Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
            :root {{
                --page-bg: #f5f7fb;
                --page-gradient: linear-gradient(180deg, #f7f9ff 0%, #eef2fb 35%, #f5f7fb 100%);
                --panel-bg: #ffffff;
                --panel-border: #e2e8f0;
                --ink-900: #0f172a;
                --ink-700: #334155;
                --ink-500: #64748b;
                --accent-600: #2563eb;
                --accent-500: #3b82f6;
                --accent-300: #60a5fa;
                --shadow-sm: 0 10px 30px rgba(15, 23, 42, 0.08);
                --shadow-lg: 0 22px 55px rgba(15, 23, 42, 0.14);
                --radius-lg: 24px;
            }}
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                padding: 2.5rem 0;
                font-family: 'Manrope', 'Segoe UI', sans-serif;
                background: var(--page-gradient);
                color: var(--ink-900);
                line-height: 1.6;
            }}
            .page-shell {{
                max-width: 1180px;
                margin: 0 auto;
                padding: 0 1.75rem 3rem;
            }}
            .page-hero {{
                background: radial-gradient(circle at top left, rgba(59, 130, 246, 0.32), transparent 62%),
                    radial-gradient(circle at 90% 15%, rgba(129, 140, 248, 0.25), transparent 65%),
                    linear-gradient(135deg, #0f172a, #1e293b 55%, #1d4ed8 130%);
                color: #fff;
                padding: 3.25rem 3rem;
                border-radius: var(--radius-lg);
                box-shadow: var(--shadow-lg);
                position: relative;
                overflow: hidden;
            }}
            .page-hero::after {{
                content: '';
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at 25% 20%, rgba(96, 165, 250, 0.38), transparent 60%);
                opacity: 0.8;
                pointer-events: none;
            }}
            .hero-inner {{
                position: relative;
                z-index: 1;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}
            .hero-eyebrow {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(15, 23, 42, 0.45);
                border: 1px solid rgba(148, 163, 184, 0.25);
                padding: 0.4rem 0.9rem;
                border-radius: 999px;
                font-size: 0.85rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .page-hero h1 {{
                margin: 0;
                font-size: 2.6rem;
                font-weight: 700;
                letter-spacing: -0.01em;
            }}
            .hero-meta {{
                display: flex;
                flex-wrap: wrap;
                gap: 1.25rem;
                font-size: 0.95rem;
                color: rgba(226, 232, 240, 0.85);
            }}
            main {{
                margin-top: 2.75rem;
                display: grid;
                gap: 2.5rem;
            }}
            section {{
                background: var(--panel-bg);
                border-radius: var(--radius-lg);
                border: 1px solid var(--panel-border);
                box-shadow: var(--shadow-sm);
                padding: 2.25rem 2.5rem;
            }}
            .section-heading {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                margin-bottom: 1.75rem;
            }}
            .section-heading h2 {{
                margin: 0;
                font-size: 1.55rem;
                font-weight: 700;
                letter-spacing: -0.01em;
            }}
            .section-heading p {{
                margin: 0;
                color: var(--ink-500);
                font-size: 0.97rem;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
                gap: 1.5rem;
            }}
            .stat-card {{
                position: relative;
                padding: 1.65rem 1.7rem;
                border-radius: 20px;
                border: 1px solid rgba(148, 163, 184, 0.25);
                background: linear-gradient(160deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.65) 100%);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6), 0 12px 32px rgba(15, 23, 42, 0.06);
                display: flex;
                align-items: center;
                gap: 1.25rem;
                overflow: hidden;
            }}
            .stat-card::after {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(99, 102, 241, 0.07));
                opacity: 0;
                transition: opacity 200ms ease;
            }}
            .stat-card:hover::after {{
                opacity: 1;
            }}
            .stat-card::before {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(160deg, var(--swatch-start), var(--swatch-end));
                opacity: 0.18;
                pointer-events: none;
            }}
            .stat-emblem {{
                position: relative;
                z-index: 1;
                width: 3.1rem;
                height: 3.1rem;
                border-radius: 50%;
                display: grid;
                place-items: center;
                background: rgba(255, 255, 255, 0.9);
                color: var(--accent-600);
                font-size: 1.3rem;
                font-weight: 600;
                box-shadow: 0 6px 18px rgba(37, 99, 235, 0.18);
            }}
            .stat-content {{
                position: relative;
                z-index: 1;
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }}
            .stat-label {{
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--ink-500);
                font-weight: 600;
            }}
            .stat-value {{
                font-size: 1.55rem;
                font-weight: 700;
                color: var(--ink-900);
            }}
            .card-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.75rem;
                margin-top: 1.5rem;
            }}
            .card {{
                background: linear-gradient(155deg, rgba(255, 255, 255, 0.98) 0%, rgba(241, 245, 249, 0.72) 100%);
                border-radius: 20px;
                border: 1px solid rgba(148, 163, 184, 0.25);
                padding: 1.75rem;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65), 0 14px 32px rgba(15, 23, 42, 0.08);
                display: flex;
                flex-direction: column;
                gap: 1.25rem;
            }}
            .table-card {{
                padding: 1.25rem;
            }}
            .graph-head {{
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                gap: 0.75rem;
            }}
            .graph-head h3 {{
                margin: 0;
                font-size: 1.05rem;
                color: var(--ink-900);
            }}
            .graph-link {{
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--accent-600);
                text-decoration: none;
                padding-bottom: 0.1rem;
                border-bottom: 1px dashed rgba(37, 99, 235, 0.4);
            }}
            .graph-link:hover {{
                color: var(--accent-500);
                border-bottom-color: rgba(37, 99, 235, 0.6);
            }}
            .graph-canvas {{
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.3);
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
            }}
            .graph-canvas img {{
                display: block;
                width: 100%;
                height: auto;
            }}
            .table-scroll {{
                overflow-x: auto;
                border-radius: 16px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.93rem;
            }}
            .styled-table thead {{
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.12));
                color: var(--ink-700);
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-size: 0.78rem;
            }}
            .styled-table th,
            .styled-table td {{
                padding: 0.85rem 1rem;
                text-align: center;
                border-bottom: 1px solid rgba(226, 232, 240, 0.8);
                white-space: nowrap;
            }}
            .styled-table tbody tr:nth-child(even) {{
                background: rgba(226, 232, 240, 0.35);
            }}
            .styled-table tbody tr:hover {{
                background: rgba(59, 130, 246, 0.12);
            }}
            .styled-table tbody tr:last-child td {{
                border-bottom: none;
            }}
            section ul {{
                margin: 0;
                padding-left: 1.2rem;
                color: var(--ink-700);
            }}
            @media (max-width: 980px) {{
                .page-hero {{
                    padding: 2.75rem 2.4rem;
                }}
                section {{
                    padding: 2rem 1.85rem;
                }}
            }}
            @media (max-width: 720px) {{
                body {{
                    padding: 1.75rem 0 2.5rem;
                }}
                .page-shell {{
                    padding: 0 1.25rem 2.25rem;
                }}
                .page-hero {{
                    border-radius: 22px;
                    padding: 2.4rem 1.9rem;
                }}
                .page-hero h1 {{
                    font-size: 2.1rem;
                }}
                main {{
                    margin-top: 2.25rem;
                    gap: 2rem;
                }}
                section {{
                    padding: 1.65rem 1.5rem;
                }}
                .stat-card {{
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.9rem;
                }}
                .stat-value {{
                    font-size: 1.45rem;
                }}
                .hero-meta {{
                    flex-direction: column;
                    gap: 0.5rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class='page-shell'>
            <header class='page-hero'>
                <div class='hero-inner'>
                    <span class='hero-eyebrow'>Futures Trading Performance</span>
                    <h1>Trade Analysis Report</h1>
                    <div class='hero-meta'>
                        <span>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                        <span>All figures quoted in USD unless stated otherwise</span>
                    </div>
                </div>
            </header>

            <main>
                <section class='summary-section'>
                    <div class='section-heading'>
                        <h2>Summary Statistics</h2>
                        <p>Headline results that frame overall trade activity and profitability for the selected window.</p>
                    </div>
                    <div class='stats-grid'>
                        {summary_html}
                    </div>
                </section>

                <section class='performance-section'>
                    <div class='section-heading'>
                        <h2>Performance Metrics</h2>
                        <p>Risk- and consistency-focused metrics that reveal the character of returns and trade management.</p>
                    </div>
                    <div class='stats-grid'>
                        {performance_html}
                    </div>
                </section>

                {trade_highlights_section}

                <section class='visual-section'>
                    <div class='section-heading'>
                        <h2>Visual Highlights</h2>
                        <p>Key charts illustrating cumulative performance, distribution of returns, and session tendencies.</p>
                    </div>
                    <div class='card-grid'>
                        {''.join(graph_cards)}
                    </div>
                </section>

                <section class='profitability-section'>
                    <div class='section-heading'>
                        <h2>Profitability by Day and Hour</h2>
                        <p>Heatmap summarising when performance clusters, highlighting both fruitful and underperforming sessions.</p>
                    </div>
                    <div class='card table-card'>
                        <div class='table-scroll'>
                            {day_hour_summary_html}
                        </div>
                    </div>
                </section>

                <section class='daily-section'>
                    <div class='section-heading'>
                        <h2>Daily Net Profit and Loss</h2>
                        <p>Daily contribution to account equity, helping isolate volatility pockets across the sample.</p>
                    </div>
                    <div class='card table-card'>
                        <div class='table-scroll'>
                            {daily_net_pnl_html}
                        </div>
                    </div>
                </section>

                <section class='distribution-section'>
                    <div class='section-heading'>
                        <h2>Trade Distribution by Profit/Loss Range</h2>
                        <p>Trade frequency by P&L bucket to visualise expectancy drivers and the depth of drawdowns.</p>
                    </div>
                    <div class='card table-card'>
                        <div class='table-scroll'>
                            {pnl_table_html}
                        </div>
                    </div>
                </section>
            </main>
        </div>
    </body>
    </html>
"""


    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write(html_content)

    return report_path

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
    try:
        filepath = select_file()
    except TclError:
        print("GUI file selection is not available. Falling back to sample data.")
        filepath = None

    if not filepath:
        sample_path = Path(__file__).resolve().parent / 'sample_data' / 'trade_log_sample.txt'
        if sample_path.exists():
            print(f"Using sample data from {sample_path}.")
            filepath = sample_path
        else:
            print("No file selected and sample data not found. Exiting.")
            return

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

        # Display the table of calculations for the first 10 trades and capture HTML for reporting
        _, trade_highlights_html = display_pnl_table(
            df_with_pnl,
            10,
            print_output=True,
            return_html=True
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path('reports') / f'trade_report_{timestamp}'

        # Plot cumulative P&L
        cumulative_pnl_path = plot_pnl(df_with_pnl, output_dir=report_dir, show=False)

        profit_summary = aggregate_profit_by_time(df_with_pnl)
        day_hour_summary_html, daily_net_pnl_html = build_profitability_tables(profit_summary, df_with_pnl)
        heatmap_path = plot_heatmap(profit_summary, output_dir=report_dir, show=False)
        pnl_distribution_path = plot_pnl_distribution(df_with_pnl, output_dir=report_dir, show=False)
        pnl_analysis_table = create_pnl_analysis_table(df_with_pnl)
        print(pnl_analysis_table)

        summary_stats = generate_summary_statistics(df_with_pnl)
        performance_metrics = calculate_performance_metrics(df_with_pnl)

        print("\nSummary Statistics:")
        print(f"  Total Closed Trades: {summary_stats['total_trades']}")
        print(f"  Trading Days Covered: {summary_stats['trading_days']}")
        print(f"  Total Profit/Loss: {format_currency(summary_stats['total_pnl'])}")
        print(f"  Average Daily P&L: {format_currency(summary_stats['average_daily_pnl'])}")
        print(f"  Best Trade: {format_currency(summary_stats['best_trade'])}")
        print(f"  Worst Trade: {format_currency(summary_stats['worst_trade'])}")

        print("\nPerformance Metrics:")
        print(f"  Expectancy (per trade): {format_currency(performance_metrics['expectancy'])}")
        print(f"  Standard Deviation of P&L: {format_currency(performance_metrics['pnl_std'])}")
        print(f"  Sharpe Ratio (per trade): {format_ratio(performance_metrics['sharpe_ratio'])}")
        print(f"  Win Rate: {format_percentage(performance_metrics['win_rate'])}")
        print(f"  Loss Rate: {format_percentage(performance_metrics['loss_rate'])}")
        print(f"  Average Win: {format_currency(performance_metrics['average_win'])}")
        print(f"  Average Loss: {format_currency(performance_metrics['average_loss'])}")
        print(f"  Profit Factor: {format_ratio(performance_metrics['profit_factor'])}")
        print(f"  Reward-to-Risk Ratio: {format_ratio(performance_metrics['reward_risk_ratio'])}")
        print(f"  Max Drawdown: {format_currency(performance_metrics['max_drawdown'])}")
        print(f"  Average Trade Duration: {format_minutes(performance_metrics['average_trade_duration_minutes'])}")
        print(f"  Median Trade Duration: {format_minutes(performance_metrics['median_trade_duration_minutes'])}")
        print(f"  Total Market Exposure: {format_hours(performance_metrics['exposure_hours'])}")

        report_path = generate_report(
            report_dir,
            summary_stats,
            performance_metrics,
            [
                ('Cumulative Profit and Loss Over Time', cumulative_pnl_path),
                ('Profitability Heatmap by Day and Hour', heatmap_path),
                ('Distribution of Trade Profit and Loss', pnl_distribution_path)
            ],
            pnl_analysis_table,
            day_hour_summary_html,
            daily_net_pnl_html,
            trade_highlights_html
        )

        print(f"Report generated at {report_path}")

    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    main()
