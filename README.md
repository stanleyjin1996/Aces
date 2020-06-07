# Team: Aces
## Members: Hepu Jin, Lingxiao Zhang
Using market regimes, changepoints and anomaly detection in QWIM

## Ideas from literature review
1. Regime Shifts in Excess Stock Return Predictability: An OOS Portfolio Analysis
   In this article, three financial ratios(DP, EP, BM ratios) are used to produce signals. HMM is used to model regime shifts. An
   interesting future expansion could examine the extent to which our results may apply to different predictors (e.g. other accounting
   ratios)
 
2. Regime-Based Versus Static Asset Allocation: Letting the Data Speak
   In thi article, log daily returns are used as a factor in HMM to model regime shifts. Two states are chosen: high volatility states and 
   low volatility states. Strategies of Stock-Bonds and Long-Short are tested using signals generated from HMM. It might be possible to
   improve performance by including economic vairables, interest rates, investor sentiment surbeys, or other indicators.

3. Detecting change points in VIX and S&P 500: A new approach to dynamic asset allocation.
   In this article, both VIX and S&P 500 returns have been used to generate signals. A non-parametric test Mood test is used to detect 
   change in scale. Possible further exploration could be using past volatility as data input to generate signals. We could also consider
   other non-parametric tests to detect change.
