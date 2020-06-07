# Team: Aces
## Members: Hepu Jin, Lingxiao Zhang
Using market regimes, changepoints and anomaly detection in QWIM

## Ideas from main references
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
   
4. Dynamic Allocation or Diversification: A Regime-Based Approach to Multiple Assets.
   In this article, MSCI World index has been used to generate signals. HMM is used to detect regime changes. In addition, more assets have
   been considered in the portfolio to account for diversification. We leave for future research to show whether the results presented in
   this article can be improved by including information from the other asset classses(ex. momentum index) in the regime-detection process.

## Ideas from regime-based investing references
1. Regime Shifts and Markov-Switching Models: Implications for Dynamic Strategies
   In this article, FX market turbulence, Equity market turbulence, Inflation, Gross National Product has been used to generate signals.
   HMM is used to detect regime shifts. Further research could be finding other macro variables that link to asset performance.

