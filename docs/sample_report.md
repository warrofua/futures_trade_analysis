# Sample Trade Analysis Report

_This report was generated using the bundled sample dataset (`sample_data/trade_log_sample.txt`). It illustrates the insights the trade analysis script surfaces without requiring access to live trading logs._

## Dataset Overview
- **Instrument focus:** Micro E-mini S&P 500 futures (MES contract months M3 & U3).
- **Time frame:** 14–18 August 2023 (converted from UTC to US/Eastern for presentation).
- **Number of fills:** 10 (5 round-trip trades).
- **Net P&L:** **$30.00** across the sample period.

### Trade-Level Summary
| Open Time (ET) | Symbol | Side | Qty | Open Price | Close Price | PnL ($) | Cumulative PnL ($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-08-14 09:30 | MESM3 | Buy | 1 | 4450.00 | 4452.00 | 10.00 | 10.00 |
| 2023-08-15 10:45 | MESM3 | Sell | 1 | 4461.00 | 4456.00 | 25.00 | 35.00 |
| 2023-08-16 08:20 | MESU3 | Buy | 2 | 4448.00 | 4443.00 | -50.00 | -15.00 |
| 2023-08-17 05:05 | MESU3 | Sell | 1 | 4439.00 | 4435.00 | 20.00 | 5.00 |
| 2023-08-18 07:15 | MESU3 | Buy | 1 | 4444.00 | 4449.00 | 25.00 | 30.00 |

*PnL calculations use the default MES tick value ($1.25) and four ticks per point, as configured in `trade_analysis_script.py`.*

## Visual Output Highlights
Although the script opens Matplotlib windows for interactive exploration, the key takeaways from those visuals for this dataset are:
- **Cumulative P&L line chart:** Starts strong, dips mid-week due to the Wednesday loss, and recovers into positive territory by Friday’s session, closing at **$30**.
- **PnL distribution histogram:** Right-skewed—four of the five trades finished profitable, but the losing trade was comparatively larger.
- **Profitability heatmap:** Shows profitable activity clustered around late morning sessions on Monday, Tuesday, and Friday, while Wednesday’s 8:00 ET block stands out as the main drawdown.

### Profitability by Day & Hour (US/Eastern)
| Day of Week | 05:00 | 06:00 | 07:00 | 08:00 | 09:00 | 10:00 | 11:00 | 12:00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Monday | 0.0 | 0.0 | 0.0 | 0.0 | **10.0** | 0.0 | 0.0 | 0.0 |
| Tuesday | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **25.0** | 0.0 | 0.0 |
| Wednesday | 0.0 | 0.0 | 0.0 | **-50.0** | 0.0 | 0.0 | 0.0 | 0.0 |
| Thursday | **20.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Friday | 0.0 | 0.0 | **25.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Saturday | — | — | — | — | — | — | — | — |
| Sunday | — | — | — | — | — | — | — | — |

### PnL Range Distribution
| PnL Range ($) | Count of Trades |
| --- | --- |
| -40 to -30 | 0 |
| -30 to -20 | 0 |
| -20 to -10 | 0 |
| -10 to -5 | 0 |
| -5 to 0 | 5 |
| 0 to 5 | 0 |
| 5 to 10 | 1 |
| 10 to 20 | 1 |
| 20 to 30 | 2 |
| 30 to 40 | 0 |
| ≤ -40 | 1 |
| > 40 | 0 |

## Observations
- The single outsized loss on Wednesday morning erased the gains from the two earlier trades, emphasizing the need for risk controls during that session.
- Best-performing windows in the sample are late morning on Tuesday and early morning on Thursday/Friday, suggesting a focus area for future strategy refinement.
- Even with one large losing trade, the strategy recovered within two sessions—highlighting the value of consistent position sizing and sticking to higher-probability setups.

Use this report as a template: rerun the script with your own Sierra Chart exports and capture the same tables/visuals to track your personal trading performance.
