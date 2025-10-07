# Sample Trade Analysis Report

_This report was generated using the bundled sample dataset (`sample_data/trade_log_sample.txt`). It illustrates the insights the trade analysis script surfaces without requiring access to live trading logs._

## Dataset Overview
- **Instrument focus:** Micro E-mini S&P 500 futures (MES contract months U3 & Z3).
- **Time frame:** 14–25 August 2023 (converted from UTC to US/Eastern for presentation).
- **Number of fills:** 40 (20 round-trip trades across two trading weeks).
- **Net P&L:** **$785.00** across the sample period.
- **Performance snapshot:**
  - Expectancy: **$39.25** per trade
  - Sharpe ratio: **11.26** (per-trade basis)
  - Win rate: **100%**
  - Average trade duration: **90.3 minutes**
  - Total market exposure: **30.1 hours**

### Trade-Level Summary
| Open Time (ET) | Close Time (ET) | Symbol | Side | Qty | Open Price | Close Price | PnL ($) | Cumulative PnL ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-08-14 05:35 | 2023-08-14 06:50 | MESU3 | Buy  | 1 | 4448.00 | 4454.00 | 30.00 | 30.00 |
| 2023-08-14 09:15 | 2023-08-14 10:45 | MESU3 | Sell | 1 | 4459.00 | 4452.00 | 35.00 | 65.00 |
| 2023-08-15 05:40 | 2023-08-15 07:05 | MESU3 | Sell | 2 | 4463.00 | 4459.00 | 40.00 | 105.00 |
| 2023-08-15 09:25 | 2023-08-15 11:10 | MESU3 | Buy  | 1 | 4455.00 | 4462.00 | 35.00 | 140.00 |
| 2023-08-16 05:32 | 2023-08-16 06:55 | MESU3 | Buy  | 1 | 4464.00 | 4470.00 | 30.00 | 170.00 |
| 2023-08-16 09:05 | 2023-08-16 10:35 | MESU3 | Sell | 1 | 4473.00 | 4466.00 | 35.00 | 205.00 |
| 2023-08-17 05:20 | 2023-08-17 06:40 | MESU3 | Buy  | 1 | 4461.00 | 4468.00 | 35.00 | 240.00 |
| 2023-08-17 09:10 | 2023-08-17 10:50 | MESU3 | Sell | 1 | 4465.00 | 4457.00 | 40.00 | 280.00 |
| 2023-08-18 05:25 | 2023-08-18 07:10 | MESU3 | Buy  | 2 | 4458.00 | 4465.00 | 70.00 | 350.00 |
| 2023-08-18 09:45 | 2023-08-18 11:20 | MESU3 | Sell | 1 | 4463.00 | 4459.00 | 20.00 | 370.00 |

*Table shows the first 10 round-trip trades from the two-week sample dataset. Subsequent trades continue the pattern during the following contract roll (MESZ3).* 

*PnL calculations use the default MES tick value ($1.25) and four ticks per point, as configured in `trade_analysis_script.py`.*

## Visual Output Highlights
Although the script opens Matplotlib windows for interactive exploration, the key takeaways from those visuals for this dataset are:
- **Cumulative P&L line chart:** Climbs steadily throughout both weeks, closing at **$785** thanks to consistent execution and moderate position sizing.
- **PnL distribution histogram:** Concentrated around the $30–$40 per-trade band, reflecting the tight dispersion of outcomes in the sample.
- **Profitability heatmap:** Highlights repeatable strength in the 05:00 ET and 09:00 ET sessions from Monday through Friday, with Friday showing the highest combined contribution.

### Profitability by Day & Hour (US/Eastern)
| Day of Week | 05:00 | 09:00 |
| --- | --- | --- |
| Monday | **65.0** | **70.0** |
| Tuesday | **70.0** | **70.0** |
| Wednesday | **100.0** | **75.0** |
| Thursday | **60.0** | **75.0** |
| Friday | **100.0** | **100.0** |
| Saturday | — | — |
| Sunday | — | — |

### PnL Range Distribution
| PnL Range ($) | Count of Trades |
| --- | --- |
| -40 to -30 | 0 |
| -30 to -20 | 0 |
| -20 to -10 | 0 |
| -10 to -5 | 0 |
| -5 to 0 | 0 |
| 0 to 5 | 0 |
| 5 to 10 | 0 |
| 10 to 20 | 1 |
| 20 to 30 | 5 |
| 30 to 40 | 11 |
| ≤ -40 | 0 |
| > 40 | 3 |

## Observations
- Twenty consecutive profitable trades delivered **$785** over ten trading days, translating to a high expectancy (**$39.25** per trade) and an institutional-grade Sharpe ratio (**11.26**) on a per-trade basis.
- Profits were spread evenly across both early-morning (05:00 ET) and mid-morning (09:00 ET) sessions, suggesting the strategy thrives when liquidity deepens after the overnight session.
- The absence of losses in this curated dataset drives an infinite profit factor; in live trading, the same analytics will quickly surface risk concentrations or drawdowns that require attention.

Use this report as a template: rerun the script with your own Sierra Chart exports and capture the same tables/visuals to track your personal trading performance.
