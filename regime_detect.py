import pandas as pd
import numpy as np
import pickle
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.decomposition import PCA


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

    def train(self, rets):
        """
        Train the hmm model
        :param rets: Returns of one specific asset (need to be ndarray)
        """
        self.model.fit(rets)
        # Update class attributes
        self.monitor_ = self.model.monitor_
        self.transmat_ = self.model.transmat_
        self.means_ = self.model.means_
        self.covars_ = self.model.covars_

    def predict(self, rets):
        """
        Predict the hidden states
        :param rets: Returns of one specific asset (need to be ndarray)
        :return: Predicted hidden states
        :rtype: array
        """
        return self.model.predict(rets)

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
        hidden_states = self.model.predict(rets)
        # Create the correctly formatted plot
        fig, axs = plt.subplots(self.n_components, sharex=True, sharey=True)
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

    def load_model(self, file_path):
        """
        Load the hmm model from a local path
        :param file_path: File path for loading
        """
        print("Loading HMM model...")
        with open(file_path, "rb") as file:
            pickle.load(file)
        print("HMM model loaded ")


class BackTest:
    """
    BackTest is a class defining the back test procedure
    """

    def __init__(self, model):
        """
        Initialize class attributes
        :param model: A model needed to be backtested
        """
        self.model = model

    def run(self):
        """
        Run the back test process
        :return:
        """
        return

    def fetch_sector(self, start, end):
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
                   "XLRE": "S&P Real Estate Select Sector (XLRE)",
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

    def __init__(self, assets_names):
        """
        Initialize class attributes
        :param assets_names: A list of assets' names
        """
        self.assets_names = assets_names
        self.weights = []
        self.assets_data = []

    def absorption_ratio(self, data, is_returns=False, lookback=500, halflife=250, num_pc=5):
        """
        Calculate the absorption ratio from
        :param data:
        :param is_returns:
        :param lookback:
        :param halflife:
        :param num_pc:
        :return:
        """
        if is_returns:
            ret = data
        else:
            ret = np.log(df_price).diff().dropna()

        ar = []
        for i in range(lookback, len(ret) + 1):
            model = PCA(n_components=num_pc).fit(ret[i - lookback:i])
            pc = pd.DataFrame(model.transform(ret))
            # variance explained by first n principal components
            pc_var = pc.ewm(halflife=halflife, min_periods=lookback).var().sum().sum()
            # variance in original data
            ret_var = ret.ewm(halflife=halflife, min_periods=lookback).var().sum().sum()
            # absorption ratio
            ar.append(pc_var / ret_var)

        ar = pd.DataFrame(ar)
        ar.index = ret.index[lookback - 1:]
        ar.columns = ['AR']

        return ar

    def optimize(self):
        return

    def visualization(self):
        return

    def sharpe_ratio(self, rp, rf, days=252):
        """
        Calculate the sharpe ratio of the portfolio
        :param rp: Daily return of the portfolio
        :param rf: Risk free rate
        :param days: Time span
        :return: Sharpe ratio of the portfolio
        :rtype: float
        """
        volatility = rp.std() * np.sqrt(days)
        sr = (rp.mean() - rf) / volatility
        return sr

    def metric(self):
        return


if __name__ == '__main__':
    model = HMM(2, 1000)
    bt = BackTest(model)
    df_price, df_return = bt.fetch_sector("2010-01-01", "2020-07-11")
    rets = np.column_stack([df_return["XLC"].iloc[1:]])
    model.train(rets)
    df = pd.DataFrame(data=df_price["XLC"].iloc[1:])
    df.columns = ["Adj Close"]
    model.plot_in_sample_hidden_states(df)
    print(model.score(rets))



