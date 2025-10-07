# Interpreting Advanced Futures Analytics

The enriched trade report surfaces a suite of risk- and distribution-aware diagnostics
so discretionary traders can turn raw fills into actionable playbooks. Use the
guidance below to translate the new widgets into execution decisions.

## Monte Carlo Scenario Bands
- **What it shows:** Bootstrap resamples of realised trade PnL build percentile equity
  curves (5th–95th). They illustrate how the same trade distribution can fan out over
  future sequences.
- **How to use it:**
  - When the 5th percentile curve trends higher, the strategy is resilient even under
    adverse sequencing; flat or negative slopes flag fragility.
  - Compare the median and 95th percentile to understand upside dispersion. Wide gaps
    suggest highly skewed payoff profiles that need tighter risk sizing.
- **Risk of ruin:** The simulation table reports the share of paths that breached the
  historical drawdown depth. Keep this below the desk’s tolerance (e.g., <5%); if it
  spikes, scale down trade size or tighten stops until the ruin probability retreats.

## Drawdown Diagnostics
- **Histogram:** Highlights how frequently equity pullbacks of different magnitudes
  occurred. A long right tail means outsized pain events are possible and should be
  budgeted for in capital planning.
- **Event table:** Lists peak-to-trough depth and trade-count duration for the deepest
  drawdowns. Use the duration metric to estimate how long recovery typically takes and
  set expectations with investors.
- **Action:** If new drawdowns exceed historical depth or persist longer than the table
  implies, pause and reassess market conditions before adding risk.

## Volatility Regime Breakdown
- **Rolling standard deviation:** Measures how noisy per-trade outcomes were over the
  latest 10-trade window (with medium and high thresholds plotted for context).
- **Regime table:** Splits performance into low, medium, and high volatility states,
  surfacing how win rate and average PnL shift.
- **Action:** Allocate more size to the regime that historically produced the healthiest
  expectancy. If high-volatility regimes erode win rate, throttle back or tighten
  targets when the rolling band breaches the "high" threshold.

## Payoff Ratio Distribution
- **Quantiles:** Show how individual winners compare with the typical loss. A p50 above
  1.0 indicates that even the median winner outpaces the average loss, supporting the
  edge claim.
- **Action:** Track the p10 and p25 levels. If they slip below 0.7–0.8, winners are no
  longer paying for losses—tighten entries or pare risk until the lower quantiles
  improve.

## Ulcer Index
- **Definition:** Root-mean-square of percentage drawdowns from equity highs. Unlike
  standard deviation it penalises only downside excursions, capturing the "stress" the
  account experiences.
- **Action:** Maintain the Ulcer index within acceptable ranges for your mandate (e.g.,
  <8% for swing strategies). Rising values warn that drawdowns are both deeper and more
  persistent; respond by trimming trade frequency or cutting leverage.

## Rolling Volatility Snapshot
- **Metrics:** Average and most recent rolling PnL volatility across 10- and 20-trade
  windows.
- **Action:** Use the latest 20-trade figure as an alert level. If it surges beyond
  historic norms, throttle back exposure until volatility mean-reverts.

---

Keep the advanced analytics in view alongside the core summary cards. Together they
provide a forward-looking lens on distribution risk, allowing you to scale position
size, refine execution timing, and survive inevitable adverse runs.
