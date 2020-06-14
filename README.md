# Team: Aces
## Members: Hepu Jin, Lingxiao Zhang
Using market regimes, changepoints and anomaly detection in QWIM

## Ideas from main references
1. Regime Shifts in Excess Stock Return Predictability: An OOS Portfolio Analysis
   In this article, three financial ratios(DP, EP, BM ratios) are used to produce signals. HMM is used to model regime shifts. **An**
   **interesting future expansion could examine the extent to which our results may apply to different predictors (e.g. other accounting**
   **ratios)**
 
2. Regime-Based Versus Static Asset Allocation: Letting the Data Speak
   In thi article, log daily returns are used as a factor in HMM to model regime shifts. Two states are chosen: high volatility states and 
   low volatility states. Strategies of Stock-Bonds and Long-Short are tested using signals generated from HMM. **It might be possible to**
   **improve performance by including economic vairables, interest rates, investor sentiment surbeys, or other indicators.**

3. Detecting change points in VIX and S&P 500: A new approach to dynamic asset allocation.
   In this article, both VIX and S&P 500 returns have been used to generate signals. A non-parametric test Mood test is used to detect 
   change in scale. **Possible further exploration could be using past volatility as data input to generate signals. We could also consider**
   **other non-parametric tests to detect change.**
   
4. Dynamic Allocation or Diversification: A Regime-Based Approach to Multiple Assets.
   In this article, MSCI World index has been used to generate signals. HMM is used to detect regime changes. In addition, more assets have
   been considered in the portfolio to account for diversification. **We leave for future research to show whether the results presented in**
   **this article can be improved by including information from the other asset classses(ex. momentum index) in the regime-detection process.**

## Ideas from regime-based investing references
1. Regime Shifts and Markov-Switching Models: Implications for Dynamic Strategies.
   In this article, FX market turbulence, Equity market turbulence, Inflation, Gross National Product has been used to generate signals.
   HMM is used to detect regime shifts. **Further research could be finding other macro variables that link to asset performance.**
2. Optimizing asset allocations to market regimes.
   In this article, market regime indicator is composed of three market factors: equity implied volatility factor, currency implied 
   volatility factor, credit spread factor. Market regimes are defined into 5 categories. Capital is allocated among growth assets, 
   moderate assets and defensive assets depending on different market regimes.
3. Dynamic Strategic Asset Allocation: Risk and Return across Economic Regimes.
   In this article, authors consider a regime model which uses four economic indicators (the credit spread, earnings yield, ISM,
   unemployment rate) to identify four phases of the economic cycle (expansion, peak, recession and recovery). More specifically,
   they standardize the foru economic variables and add them together and divide by the square root of 4. **It turns out that they** 
   **assume these four variables are normally distributed and uncorrelated. However, the factors are actually positively correlated**
   **especially during stressful period. Further improvement could be finding a remedy for this.**
5. Regime Shifts: Implications for Dynamic Strategies. The authos show how to apply Markov-switching models to forecast regimes in
   market turbulance, inflation, economic growth. Details regarding the model and algorithm are provided.

## Some topics to be considered for the project
1. **Using different unsupervised learning algorithms to detect regime shifts. For example, K-means clustering, Mean-shift clustering,**
   **DBSCAN, EM clustering using GMM.**

2. **Using different supervised learning algorithms to detect regime shifts. Features are economic vairables (the credit spread, earnings**
   **yield, ISM, unemployment rate) and target is NBER indicator.**
