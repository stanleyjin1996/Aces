import pandas as pd
import numpy a np
import yfinance as yf
from sklearn.decomposition import PCA


def fetch_sector(start, end):
    """
    Get 11 sectors' data
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
        df_return[k] = sector_data[k]["Close"].pct_change()
        df_price[k] = sector_data[k]["Close"]

    return df_price, df_return


df_price, df_return = fetch_sector("2010-01-01", "2020-07-11")

#Absorption Ratio
def absorption_ratio(data,is_returns=False,lookback=500,halflife=250,num_pc=5):
    if is_returns:
        ret = data
    else:
        ret = np.log(df_price).diff().dropna()
    
    ar = []
    for i in range(lookback,len(ret)+1):
        model = PCA(n_components=num_pc).fit(ret[i-lookback:i])
        pc = pd.DataFrame(model.transform(ret))
        #variance explained by first n principal components
        pc_var = pc.ewm(halflife=halflife,min_periods=lookback).var().sum().sum()
        #variance in original data
        ret_var = ret.ewm(halflife=halflife,min_periods=lookback).var().sum().sum()
        #absorption ratio
        ar.append(pc_var/ret_var)
        
    ar = pd.DataFrame(ar)
    ar.index = ret.index[lookback-1:]
    ar.columns = ['AR']
    
    return ar
ar = absorption_ratio(df_price,False,250,125,4)
ar.plot()
