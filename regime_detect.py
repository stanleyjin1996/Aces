import pandas as pd
import numpy as np
import pickle
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.decomposition import PCA
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

    def __init__(self, asset_price, tickers):
        '''
        Initialize class attributes
        :param assets_price: dataframe including trading universe and regime
        :param tickers: asset tickers
        '''
        
        self.tickers = tickers
        self.price = asset_price
        self.ret = None #daily log return
        
        self.weights_special = [] #optimum weights for regime-based portfolio
        self.weights_base = [] #optimum weights for base portfolio
        
        
        self.date0 = None #dates for regime 0
        self.date1 = None #dates for regime 1
        
        self.mu_special = None #expected return for special portfolio
        self.S_special = None #covariance matrix for special portfolio
        
        self.mu_base = None #expected return for base portfolio
        self.S_base = None #expected return for base portfolio
        
        self.value_special = 0 #special portfolio value
        self.value_base = 0 #base portfolio value

    def initialize_ret_cov(self,n=500):
        '''
        Initialize expected returns and covariance matrices
        :param n: first n days for initializing return and cov
        '''
        self.ret = np.log(self.price.iloc[:,:-1]).diff().dropna()
        state = self.price['regime'][1:]
        df = self.ret[:n]
        #expected returns and covariance matrix for base portfolio
        self.S_base = df.ewm(span=len(df),min_periods=len(df)).cov().dropna() * 252
        self.mu_base = df.ewm(span=len(df),min_periods=len(df)).mean().iloc[-1] * 252
        #expected returns and covariance matrix for special portfolio
        self.date0 = list(state[:n][state == 0].index)
        self.date1 = list(state[:n][state == 1].index)
        if state[n] == 0:
            df = df.loc[self.date0]
        else:
            df = df.loc[self.date1]
        self.S_special = df.ewm(span=len(df),min_periods=len(df)).cov().dropna() * 252
        self.mu_special = df.ewm(span=len(df),min_periods=len(df)).mean().iloc[-1] * 252 
    
    def UpdateRetCov(self,state):
        '''
        Update expected returns and covairance matrices for base and special portfolios
        :param state: today's regime: 0 or 1
        '''
        #special portfolio
        if state == 0:
            span = len(self.date0)
            self.mu_special = self.ret.loc[self.date0].ewm(span=span,min_periods=span).mean().iloc[-1] * 252
            self.S_special = self.ret.loc[self.date0].ewm(span=span,min_periods=span).cov().dropna() * 252
        elif state == 1:
            span = len(self.date1)
            self.mu_special = self.ret.loc[self.date1].ewm(span=span,min_periods=span).mean().iloc[-1] * 252
            self.S_special = self.ret.loc[self.date1].ewm(span=span,min_periods=span).cov().dropna() * 252
        #base portfolio
        span = len(self.date0) + len(self.date1)
        date = sorted(self.date0 + self.date1)
        self.mu_base = self.ret.loc[date].ewm(span=span,min_periods=span).mean().iloc[-1] * 252
        self.S_base = self.ret.loc[date].ewm(span=span,min_periods=span).cov().dropna() * 252
            
    def to_list(self,dic):
        '''
        convert a OrderedDict into a list
        :param dic: ordered dictionary
        '''
        return [dic[i] for i in self.tickers]
    
    def OptimizeWeight(self,n=500):
        '''
        Choose optimum weights for each day
        :param n: start computing weights after n days
        '''
        state = self.price['regime'][1:] 
        #initialize expected returns and covariance matrix
        self.initialize_ret_cov(n)
        #determine first weights for special portfolio
        ef = EfficientFrontier(self.mu_special, self.S_special, weight_bounds=(-1,1))
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=0.02)
        #special weight
        special = to_list(ef.clean_weights())
        self.weights_special.append(special)
        #determine first weights for baseline portfolio
        ef = EfficientFrontier(self.mu_base, self.S_base, weight_bounds=(-1,1))
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=0.02)
        #base weight
        base = to_list(ef.clean_weights())
        self.weights_base.append(base)
        for i,t in zip(range(n,len(state)),self.ret.index[n:]):
            if state[i] == 0:
                self.date0.append(t)
            else:
                self.date1.append(t)
            
            #no regime shift
            if state[i] == state[i-1]:
                self.weights_special.append(special)
                self.weights_base.append(base)
            #regime shift
            else:
                self.UpdateRetCov(state=state[i])
                #update weights for special portfolio
                ef = EfficientFrontier(self.mu_special, self.S_special, weight_bounds=(-1,1))
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.max_sharpe(risk_free_rate=0.02)
                special = to_list(ef.clean_weights())
                #update weights for baseline portfolio
                ef = EfficientFrontier(self.mu_base, self.S_base, weight_bounds=(-1,1))
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.max_sharpe(risk_free_rate=0.02)
                base = to_list(ef.clean_weights())
                #store weights
                self.weights_special.append(special)     
                self.weights_base.append(base)  
        
        date = self.ret.index[n-1:]
        self.weights_special = pd.DataFrame(self.weights_special,columns=self.tickers,index=date)
        self.weights_base = pd.DataFrame(self.weights_special,columns=self.tickers,index=date)

    def Portfolio(self,capital=1e6,rebalance=60):
        '''
        Compute special/base portfolio value for each day
        :param capital: starting capital to invest
        :param rebalance: rebalance frequency in days
        '''
        #compute optimum weights
        self.OptimizeWeight(n=500)
        #retrieve price and regime data
        price = self.price.loc[self.weights_special.index].drop('regime',axis=1)
        regime = self.price.loc[self.weights_special.index]['regime']
        #portfolio values
        special = []
        base = []
        for i,t in zip(range(len(price)),price.index):
            #when regime shifts, reallocate capital
            if i > 0 and regime.iloc[i] != regime.iloc[i-1]:
                holdings1 = special[-1] * self.weights_special.loc[t] / price.loc[t]
                holdings2 = base[-1] * self.weights_base.loc[t] / price.loc[t]
            
            #rebalance the portoflios regularly
            elif i > 0 and i % rebalance == 0:
                holdings1 = special[-1] * self.weights_special.loc[t] / price.loc[t]
                holdings2 = base[-1] * self.weights_base.loc[t] / price.loc[t]
            
            elif i == 0:
                holdings1 = capital * self.weights_special.loc[t] / price.loc[t]
                holdings2 = capital * self.weights_base.loc[t] / price.loc[t]
            
            value1 = np.sum(holdings1 * price.loc[t]) #special
            value2 = np.sum(holdings2 * price.loc[t]) #base
            
            special.append(value1)
            base.append(value2)
        
        self.value_special = pd.DataFrame(special,columns=['value'],index=price.index)
        self.value_base = pd.DataFrame(special,columns=['value'],index=price.index)

    def visualization(self):
        '''
        Plot and compare special and base portoflios
        
        '''
        df = pd.DataFrame()
        df['special'] = self.value_special['value']
        df['base'] = self.value_base['value']
        df.plot()


    def metric(self,portfolio):
        '''
        Evaluate portfolio performance
        :param portfolio: dataframe of portfolio value
        
        return annual return, volatility, Sharpe ratio, Maximum Drawdown
        '''
        df = portfolio.copy()
        df['year'] = [df.index[i].year for i in range(len(df))]
        metric = pd.DataFrame()
        
        #annualized return
        annual_ret = (df.groupby('year').last() - df.groupby('year').first())/df.groupby('year').first()
        days = df.groupby('year').count()
        annual_ret = (1 + annual_ret)**(252/days) - 1
        metric = annual_ret
        metric['days'] = days
        metric.columns = ['annual return','days']
        
        #annualilzed volatility
        df['daily return'] = df['value'].pct_change()
        metric['daily vol'] = df.groupby('year')[['daily return']].std()
        metric['annual vol'] = metric['daily vol'] * np.sqrt(metric['days'])
        
        #sharpe ratio
        metric['Sharpe'] = metric['annual return']/metric['annual vol']
        
        #MDD
        df['cummax'] = df.groupby('year')['value'].cummax()
        df['DD'] = (df['value'] - df['cummax'])/df['cummax']
        metric['MDD'] = df.groupby('year')['DD'].min()
        
        return metric[['annual return','annual vol','Sharpe','MDD']]


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



