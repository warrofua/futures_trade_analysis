# Futures Trade Analysis

This repository contains a Python-based analytics workflow for exploring Sierra Chart futures trade logs (configured for Micro E-mini S&P 500 futures - MES - by default). The tool builds profit-and-loss summaries, a profitability heatmap, and distribution visualizations to help evaluate trading performance across time.

## Key Features
- **Cumulative P&L tracking** – visualize how profits evolve over the trading period.
- **Day/hour profitability heatmap** – surface the sessions that historically worked best (or worst) for you.
- **PnL distribution histogram** – highlight the skew of winning versus losing trades.
- **PnL range summary table** – break trades into performance buckets for rapid review.
- **Sample dataset & report** – included so you can explore the workflow without exporting your own log first.

## Quick Start
1. **Install dependencies** (preferably inside a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the analysis script**:
   ```bash
   python trade_analysis_script.py
   ```
3. **Select a Sierra Chart trade log** when prompted (tab-separated `.txt`). The script will transform the timestamps to US/Eastern, calculate trade PnL, and display:
   - a cumulative P&L line chart,
   - a profitability heatmap by day of week and hour,
   - a PnL distribution histogram, and
   - tabular summaries in the terminal.

> **Tip:** If you are testing the project, choose the bundled sample file at `sample_data/trade_log_sample.txt` when the file dialog opens.

## Input Data Expectations
The script expects a tab-separated text file with at least the following Sierra Chart columns:

| Column | Description |
| --- | --- |
| `TransDateTime` | UTC timestamp of the execution (will be localized to US/Eastern). |
| `Symbol` | Instrument symbol (e.g., `MESU3`). |
| `Quantity` | Number of contracts. |
| `BuySell` | Direction of the fill (`buy`/`sell`). |
| `FillPrice` | Execution price scaled by 100 (Sierra Chart default for MES). |
| `OpenClose` | Whether the fill opens or closes a position. |
| `HighDuringPosition` | Highest price reached while the position was open (scaled by 100). |
| `LowDuringPosition` | Lowest price reached while the position was open (scaled by 100). |

Trades in other instruments can be analyzed by adjusting the `tick_value` and `ticks_per_point` constants in `calculate_trade_pnl` inside `trade_analysis_script.py`.

## Sample Data & Report
A comprehensive sample dataset is provided under [`sample_data/trade_log_sample.txt`](sample_data/trade_log_sample.txt). The companion [sample report](docs/sample_report.md) demonstrates what the analysis outputs look like when run against that file, including:

- the cumulative P&L progression,
- day/hour profitability insights,
- PnL distribution buckets, and
- key takeaways observed from the trades.

Use the sample dataset to familiarize yourself with the workflow or to validate new features with predictable results.

## Customization Ideas
- Swap in your own tick values or add symbol-specific configurations.
- Extend the plotting routines to save images to disk for use in journals.
- Automate report creation (e.g., export aggregated tables to Excel or Markdown).

Contributions and ideas for enhancing the analysis are welcome—open a pull request or issue to get started!
