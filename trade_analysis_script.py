import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Function to select file
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    return file_path

# Function to preprocess data
def preprocess_and_calculate_pnl(filepath):
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
    df = pd.read_csv(filepath, sep='\t', usecols=usecols)
    df['FillPrice'] /= 100
    df['HighDuringPosition'] /= 100
    df['LowDuringPosition'] /= 100
    df = df[df['Symbol'].str.contains('/MES')]

    # Calculating P&L for closed positions only
    df['PnL'] = np.nan
    df.loc[df['OpenClose'] == 'C', 'PnL'] = (df['HighDuringPosition'] - df['LowDuringPosition']) * df['Quantity'] * 4 * 1.25
    df['TransDateTime'] = pd.to_datetime(df['TransDateTime'])
    df.sort_values(by='TransDateTime', inplace=True)
    df['CumulativePnL'] = df['PnL'].cumsum()
    
    return df

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Button('Select File', id='load-file', n_clicks=0),
    dcc.Graph(id='pnl-chart')
])

# Callback to update chart
@app.callback(
    Output('pnl-chart', 'figure'),
    [Input('load-file', 'n_clicks')]
)
def update_chart(n_clicks):
    if n_clicks > 0:
        filepath = select_file()
        if filepath:
            df = preprocess_and_calculate_pnl(filepath)
            fig = go.Figure(data=[go.Scatter(x=df['TransDateTime'], y=df['CumulativePnL'], mode='lines+markers')])
            fig.update_layout(title='Cumulative P&L Over Time', xaxis_title='Time', yaxis_title='Cumulative P&L')
            return fig
    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)