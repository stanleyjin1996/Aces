import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from pypfopt import EfficientFrontier
from pypfopt import objective_functions


class HMM:
    """
    HMM is a class defining the Hidden Markov Model algorithm and its corresponding functions.
    """

    def __init__(self, n_components, n_iter):
        """
        Initialize class attributes
        :param n_components: Number of hidden states
        :param n_iter: Number of iteration for expectation maximization algorithm (EM)
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = GaussianHMM(n_components=self.n_components, n_iter=self.n_iter, covariance_type="full")
        self.monitor_ = []  # Monitor object used to check the convergence of EM
        self.transmat_ = []  # Matrix of transition probabilities between states
        self.means_ = []  # Mean parameters for each state
        self.covars_ = []  # Covariance parameters for each state
        return

    @staticmethod
    def to_numpy_array(df):
        """
        Convert panda data frame into numpy array
        :param df: A data frame needed to be converted into numpy array
        :return: A converted numpy array
        :rtype: numpy.ndarray
        """
        return np.column_stack([df])

    def train(self, rets):
        """
        Train the hmm model
        :param rets: Returns of one specific index
        """
        np_rets = self.to_numpy_array(rets)
        self.model.fit(np_rets)
        # Update class attributes
        self.monitor_ = self.model.monitor_
        self.transmat_ = self.model.transmat_
        self.means_ = self.model.means_
        self.covars_ = self.model.covars_

    def predict(self, rets):
        """
        Predict the hidden states
        :param rets: Returns of one specific index
        :return: Predicted hidden states
        :rtype: array
        """
        np_rets = self.to_numpy_array(rets)
        return self.model.predict(np_rets)

    def predict_prob(self, rets):
        """
        Compute the posterior probability for each state in the model
        :param rets: Returns of one specific index
        :return: posteriors-state-membership probabilities for each sample
        :rtype: array, shape (n_samples, n_components)
        """

        np_rets = self.to_numpy_array(rets)
        return self.model.predict_proba(np_rets)

    def score(self, rets):
        """
        Compute the log probability under the model
        :param rets: Returns of one specific asset (need to be ndarray)
        :return: Log likelihood
        :rtype: float
        """
        return self.model.score(rets)

    def plot_in_sample_hidden_states(self, df):
        """
        Plot the adjusted closing prices masked by the in-sample hidden states as a mechanism
        to understand the market regimes.
        :param df: Data frame of one asset including "Adj Close" column
        """
        # Predict the hidden states array
        # hidden_states = self.model.predict(rets)
        hidden_states = df["regime"]
        # Create the correctly formatted plot
        fig, axs = plt.subplots(self.n_components, sharex="all", sharey="none")
        colours = cm.rainbow(np.linspace(0, 1, self.n_components))

        for i, (ax, colour) in enumerate(zip(axs, colours)):
            mask = hidden_states == i
            ax.plot_date(df.index[mask], df["Adj Close"][mask], ".", linestyle='none', c=colour)
            ax.set_title("Hidden State #%s" % i)
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.grid(True)
        plt.show()

    def save_model(self, file_path):
        """
        Save the hmm model to a local path
        :param file_path: File path for saving
        """
        print("Pickling HMM model...")
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)
        print("HMM model pickled.")

    @staticmethod
    def load_model(file_path):
        """
        Load the hmm model from a local path
        :param file_path: File path for loading
        """
        print("Loading HMM model...")
        with open(file_path, "rb") as file:
            pickle.load(file)
        print("HMM model loaded ")


# No look into future, train and predict
# Adjust weight every time
class BackTest:
    """
    BackTest is a class defining the back test procedure
    """

    def __init__(self, model, portfolio):
        """
        Initialize class attributes
        :param model: A model needed to be backtested
        :param portfolio: A tested portfolio
        """
        self.model = model
        self.portfolio = portfolio
        self.regime = pd.DataFrame()
        self.switch_points = []

    def periodic_feed(self, data, init_t, days_to_rebalance, days_to_train):
        """
        Periodically feed in new data, re-estimate HMM's parameters and predict regimes
        :param data: Data frame for detecting market regimes
        :param init_t: Initial t days to initialize the model
        :param days_to_rebalance: Number of days to re-balance and re-estimate the model periodically
        :param days_to_train: How many days of data to use to train the model
        :return: Market regimes for each date
        :rtypeL: DataFrame
        """

        # Use initial t days data to initialize the hmm model
        num_observations = data.shape[0]
        train_data = data.iloc[:init_t].copy()
        self.model.train(train_data)

        # Define a DataFrame to save the regime states
        self.regime = pd.DataFrame(data=0, columns=["regime"], index=data.index)
        self.regime.loc[:init_t, "regime"] = self.model.predict(train_data)

        # A loop to feed in the new data everyday and re-estimate the model periodically
        for t in range(init_t, num_observations):
            # Re-estimate the model
            if (t - init_t)%days_to_rebalance == 0:
                train_data = data.iloc[t - days_to_train:t].copy()
                self.model.train(train_data)
                states = self.model.predict(train_data.iloc[days_to_train - days_to_rebalance:])
                self.regime.loc[t - days_to_rebalance:t, "regime"] = states

        return self.regime

    def online_step_algorithm(self, data):
        """
        Run the online_step_algorithm
        Reference: Dynamic Allocation or Diversification: A Regime-Based Approach to Multiple Assets
        :param data: Index data from model training
        :return: The regime states
        :rtype: DataFrame
        """

        # Use initial t days data to initialize the hmm model
        init_t = 252*2
        num_observations = data.shape[0]
        train_data = data.iloc[:init_t].copy()
        self.model.train(train_data)

        # TODO
        # Initialize the asset allocation based on initial regime inference
        """
        self.portfolio.initialize_ret_cov(init_t)
        portfolio_regime = self.model.predict(train_data)[-1]
        """

        # Define a confidence threshold to trade off between accuracy and latency
        threshold = 1 - 1/num_observations
        # Define a DataFrame to save the regime states
        self.regime = pd.DataFrame(data=0, columns=["regime"], index=data.index)
        self.regime.loc[:init_t - 1, "regime"] = self.model.predict(train_data.iloc[:init_t - 1])
        # Initialize looping variables
        # t is current observation
        # Total T observations are currently included
        t = init_t
        T = init_t

        # Start the online step algorithm loop
        while T < num_observations:
            print("Working on day", T, "observation")
            # Estimate the posterior probability and regime state on day t based on T observations
            post_prob = self.model.predict_prob(train_data)
            post_prob_t = np.amax(post_prob[-1])
            state = np.where(post_prob[-1] == np.amax(post_prob[-1]))

            # Update the regime state on day t if posterior probability is larger than the confidence threshhold
            # Otherwise include one more observation
            if post_prob_t > threshold:
                temp_regime = list(self.regime['regime'])
                temp_regime[t] = state[0][0]
                self.regime["regime"] = temp_regime
                # TODO
                # Update asset allocation
                """
                if portfolio_regime != state[0][0]:
                    portfolio_regime = state[0][0]
                    self.portfolio.update_ret_cov(portfolio_regime)
                """
                t = t + 1
                T = max(t, T)
            else:
                T = T + 1

            # The HMM model is re-estimated using new included data
            train_data = data.iloc[:T].copy()
            self.model.train(train_data)

        return self.regime

    def find_switch_points(self):
        """
        Find date when regime switches
        :return: A list of dates of switch points
        :rtype: list
        """

        # Initialize the list
        self.switch_points = []
        # Save the last day's state
        prev_state = self.regime.iloc[0]["regime"]

        # Loop over all the dates
        # If today's state is different from that of yesterday, then save the switch date
        for index, row in self.regime.iterrows():
            if prev_state != row["regime"]:
                self.switch_points.append(index)
            prev_state = row["regime"]

        return self.switch_points

    @staticmethod
    def fetch_sector(start, end):
        """
        Get 11 sectors' ETF data
        :param start: A start date "year-month-day"
        :param end: An end date "year-month-day"
        :return: Return the combined price and return data frames
        :rtype: DataFrame, DataFrame
        """

        # Download data of 11 sectors
        sectors = {"XLC": "S&P Communication Services Select Sector (XLC)",
                   "XLY": "S&P Consumer Discretionary Select Sector (XLY)",
                   "XLP": "S&P Consumer Staples Select Sector (XLP)",
                   "XLE": "S&P Energy Select Sector (XLE)",
                   "XLF": "S&P Financial Select Sector (XLF)",
                   "XLV": "S&P Health Care Select Sector (XLV)",
                   "XLI": "S&P Industrial Select Sector (XLI)",
                   "XLB": "S&P Materials Select Sector (XLB)",
                   "XLK": "S&P Technology Select Sector (XLK)",
                   "XLU": "S&P Utilities Select Sector (XLU)"}
        sector_data = {}

        for s in sectors.keys():
            sector_data[s] = yf.download(s, start=start, end=end)

        # Extract the close price of each sector, calculate return and put them together
        df_return = pd.DataFrame()
        df_price = pd.DataFrame()
        for k in sector_data.keys():
            df_return[k] = sector_data[k]["Adj Close"].pct_change()
            df_price[k] = sector_data[k]["Adj Close"]

        return df_price, df_return


class Portfolio:
    """
    Portfolio is a class defining the characteristics of a portfolio
    """

    def __init__(self, asset_price, tickers):
        """
        Initialize class attributes
        :param asset_price: dataframe including trading universe and regime
        :param tickers: asset tickers
        """

        self.tickers = tickers
        self.price = asset_price
        self.ret = None  # daily log return

        self.weights_special = []  # optimum weights for regime-based portfolio
        self.weights_base = []  # optimum weights for base portfolio

        self.date0 = None  # dates for regime 0
        self.date1 = None  # dates for regime 1

        self.mu_special = None  # expected return for special portfolio
        self.S_special = None  # covariance matrix for special portfolio

        self.mu_base = None  # expected return for base portfolio
        self.S_base = None  # expected return for base portfolio

        self.value_special = 0  # special portfolio value
        self.value_base = 0  # base portfolio value

    def absorption_ratio(self, lookback=500, halflife=250, num_pc=5):
        ar = []
        for i in range(lookback, len(self.ret) + 1):
            model = PCA(n_components=num_pc).fit(self.ret[i - lookback:i])
            pc = pd.DataFrame(model.transform(self.ret))
            # variance explained by first n principal components
            pc_var = pc.ewm(halflife=halflife, min_periods=lookback).var().sum().sum()
            # variance in original data
            ret_var = self.ret.ewm(halflife=halflife, min_periods=lookback).var().sum().sum()
            # absorption ratio
            ar.append(pc_var / ret_var)

        ar = pd.DataFrame(ar)
        ar.index = self.ret.index[lookback - 1:]
        ar.columns = ['AR']

        return ar

    def initialize_ret_cov(self, n=500):
        """
        Initialize expected returns and covariance matrices
        :param n: first n days for initializing return and cov
        """

        self.ret = np.log(self.price.iloc[:, :-1]).diff().dropna()
        state = self.price['regime'][1:]
        df = self.ret[:n]
        # expected returns and covariance matrix for base portfolio
        self.S_base = df.ewm(span=len(df), min_periods=len(df)).cov().dropna() * 252
        self.mu_base = df.ewm(span=len(df), min_periods=len(df)).mean().iloc[-1] * 252
        # expected returns and covariance matrix for special portfolio
        self.date0 = list(state[:n][state == 0].index)
        self.date1 = list(state[:n][state == 1].index)
        if state[n] == 0:
            df = df.loc[self.date0]
        else:
            df = df.loc[self.date1]
        self.S_special = df.ewm(span=len(df), min_periods=len(df)).cov().dropna() * 252
        self.mu_special = df.ewm(span=len(df), min_periods=len(df)).mean().iloc[-1] * 252

    def update_ret_cov(self, state):
        """
        Update expected returns and covariance matrices for base and special portfolios
        :param state: today's regime: 0 or 1
        """

        # special portfolio
        if state == 0:
            span = len(self.date0)
            self.mu_special = self.ret.loc[self.date0].ewm(span=span, min_periods=span).mean().iloc[-1] * 252
            self.S_special = self.ret.loc[self.date0].ewm(span=span, min_periods=span).cov().dropna() * 252
        elif state == 1:
            span = len(self.date1)
            self.mu_special = self.ret.loc[self.date1].ewm(span=span, min_periods=span).mean().iloc[-1] * 252
            self.S_special = self.ret.loc[self.date1].ewm(span=span, min_periods=span).cov().dropna() * 252
        # base portfolio
        span = len(self.date0) + len(self.date1)
        date = sorted(self.date0 + self.date1)
        self.mu_base = self.ret.loc[date].ewm(span=span, min_periods=span).mean().iloc[-1] * 252
        self.S_base = self.ret.loc[date].ewm(span=span, min_periods=span).cov().dropna() * 252

    def to_list(self, dic):
        """
        convert a OrderedDict into a list
        :param dic: ordered dictionary
        """
        return [dic[i] for i in self.tickers]

    def optimize_weight(self, n=500):
        """
        Choose optimum weights for each day
        :param n: start computing weights after n days
        """

        state = self.price['regime'][1:]
        # initialize expected returns and covariance matrix
        self.initialize_ret_cov(n)
        # determine first weights for special portfolio
        ef = EfficientFrontier(self.mu_special, self.S_special, weight_bounds=(-1, 1))
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=0.02)
        # special weight
        special = self.to_list(ef.clean_weights())
        self.weights_special.append(special)
        # determine first weights for baseline portfolio
        ef = EfficientFrontier(self.mu_base, self.S_base, weight_bounds=(-1, 1))
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=0.02)
        # base weight
        base = self.to_list(ef.clean_weights())
        self.weights_base.append(base)
        for i, t in zip(range(n, len(state)), self.ret.index[n:]):
            if state[i] == 0:
                self.date0.append(t)
            else:
                self.date1.append(t)

            # no regime shift
            if state[i] == state[i - 1]:
                self.weights_special.append(special)
                self.weights_base.append(base)
            # regime shift
            else:
                self.update_ret_cov(state=state[i])
                # update weights for special portfolio
                ef = EfficientFrontier(self.mu_special, self.S_special, weight_bounds=(-1, 1))
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.max_sharpe(risk_free_rate=0.02)
                special = self.to_list(ef.clean_weights())
                # update weights for baseline portfolio
                ef = EfficientFrontier(self.mu_base, self.S_base, weight_bounds=(-1, 1))
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.max_sharpe(risk_free_rate=0.02)
                base = self.to_list(ef.clean_weights())
                # store weights
                self.weights_special.append(special)
                self.weights_base.append(base)

        date = self.ret.index[n - 1:]
        self.weights_special = pd.DataFrame(self.weights_special, columns=self.tickers, index=date)
        self.weights_base = pd.DataFrame(self.weights_special, columns=self.tickers, index=date)

    def construct_portfolio(self, capital=1e6, rebalance=60):
        """
        Compute special/base portfolio value for each day
        :param capital: starting capital to invest
        :param rebalance: rebalance frequency in days
        """

        # compute optimum weights
        self.optimize_weight(n=500)
        # retrieve price and regime data
        price = self.price.loc[self.weights_special.index].drop('regime', axis=1)
        regime = self.price.loc[self.weights_special.index]['regime']
        # portfolio values
        special = []
        base = []
        for i, t in zip(range(len(price)), price.index):
            # when regime shifts, reallocate capital
            if i > 0 and regime.iloc[i] != regime.iloc[i - 1]:
                holdings1 = special[-1] * self.weights_special.loc[t] / price.loc[t]
                holdings2 = base[-1] * self.weights_base.loc[t] / price.loc[t]

            # rebalance the portoflios regularly
            elif i > 0 and i % rebalance == 0:
                holdings1 = special[-1] * self.weights_special.loc[t] / price.loc[t]
                holdings2 = base[-1] * self.weights_base.loc[t] / price.loc[t]

            elif i == 0:
                holdings1 = capital * self.weights_special.loc[t] / price.loc[t]
                holdings2 = capital * self.weights_base.loc[t] / price.loc[t]

            value1 = np.sum(holdings1 * price.loc[t])  # special
            value2 = np.sum(holdings2 * price.loc[t])  # base

            special.append(value1)
            base.append(value2)

        self.value_special = pd.DataFrame(special, columns=['value'], index=price.index)
        self.value_base = pd.DataFrame(special, columns=['value'], index=price.index)

    def visualization(self):
        """
        Plot and compare special and base portfolios
        """

        df = pd.DataFrame()
        df['special'] = self.value_special['value']
        df['base'] = self.value_base['value']
        df.plot()

    @staticmethod
    def cal_metric(portfolio):
        """
        Evaluate portfolio performance
        :param portfolio: dataframe of portfolio value
        :return: annual return, volatility, Sharpe ratio, Maximum Drawdown
        """

        df = portfolio.copy()
        df['year'] = [df.index[i].split('/')[2] for i in range(len(df))]
        metric = pd.DataFrame()

        # annualized return
        annual_ret = (df.groupby('year').last() - df.groupby('year').first()) / df.groupby('year').first()
        days = df.groupby('year').count()
        annual_ret = (1 + annual_ret) ** (252 / days) - 1
        metric = annual_ret
        metric['days'] = days
        metric.columns = ['annual return', 'days']

        # annualilzed volatility
        df['daily return'] = df['value'].pct_change()
        metric['daily vol'] = df.groupby('year')[['daily return']].std()
        metric['annual vol'] = metric['daily vol'] * np.sqrt(metric['days'])

        # sharpe ratio
        metric['Sharpe'] = metric['annual return'] / metric['annual vol']

        # MDD
        df['cummax'] = df.groupby('year')['value'].cummax()
        df['DD'] = (df['value'] - df['cummax']) / df['cummax']
        metric['MDD'] = df.groupby('year')['DD'].min()

        return metric[['annual return', 'annual vol', 'Sharpe', 'MDD']]


def main():
    model = HMM(2, 1000)
    assets = ["A", "AA", "AAPL", "ABC", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"]
    stocks = yf.download(tickers=assets, start="2000-01-27", end="2020-07-20")
    data = stocks["Adj Close"].iloc[2:]
    spy = yf.download(tickers="SPY", start="2000-01-27", end="2010-07-20")["Adj Close"]
    p = Portfolio(data, assets)
    bt = BackTest(model, p)
    rets = spy.pct_change().dropna()
    regime_data = bt.online_step_algorithm(rets)
    print(regime_data)


    #df_price, df_return = bt.fetch_sector("2010-01-01", "2020-07-11")
    #rets = np.column_stack([df_return["XLC"].iloc[1:]])
    #model.train(rets)
    #df = pd.DataFrame(data=df_price["XLC"].iloc[1:])
    #df.columns = ["Adj Close"]
    #model.plot_in_sample_hidden_states(df, rets)
    #print(model.score(rets))


if __name__ == '__main__':
    #main()
    model = HMM(2, 1000)
    assets = ["A", "AA", "AAPL", "ABC", "ABT", "ADBE", "ADI", "ADM", "ADP", "ADSK"]
    stocks = yf.download(tickers=assets, start="2000-01-27", end="2020-07-20")
    data = stocks["Adj Close"].iloc[2:]
    spy = yf.download(tickers="SPY", start="2000-01-27", end="2004-07-20")["Adj Close"]
    p = Portfolio(data, assets)
    bt = BackTest(model, p)
    rets = spy.pct_change().dropna()
    regime_data = bt.online_step_algorithm(rets)
    regime_data["Adj Close"] = spy.loc[regime_data.index]
    bt.model.plot_in_sample_hidden_states(regime_data)
    print(bt.find_switch_points())


