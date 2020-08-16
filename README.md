# Team: Aces
## Members: Hepu Jin, Lingxiao Zhang
Using market regimes, changepoints and anomaly detection in QWIM

## Project Summary
The purpose of this project is to overcome the challenge that changing market conditions present to traditional portfolio optimization. We formulate a Hidden Markov Model to detect market regimes using standardized absorption ratio, a market risk indicator, calculated by 10 MSCI U.S. sector indices. In this way, we can estimate the expected returns and corresponding covariance matrix of assets based on regimes. By design, these two parameters are calibrated to better describe the properties of different market regimes. Then, these regime-based parameters serve as the inputs of different portfolio weight optimizers, thereby constructing regime-dependent portfolios. In an asset universe consisting of U.S. equities, U.S. bonds and commodities, it is shown that regime-based portfolios have better returns, lower volatility and less tail risks compared with other competing portfolios.
