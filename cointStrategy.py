from DataManager import DataManager
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import pandas as pd
from cointAnalysis import CointAnalysis
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils import add_time, subtract_time
import numpy as np
import statsmodels.api as sm


class PairOutput:
    def __init__(self):
        """
        This class should contain all the information backtester will need to perform backtest, as is sometimes more
        than just signals. For example in this strategy it also needs the calculated beta value, mean and std for the
        previous 2 years tested
        """
        self.past_mean = None
        self.std = None
        # This dataframe holds the signals, the close values and prices etc
        self.beta = None
        self.df = None


class CointStrategy:

    def __init__(self, portfolio, params):
        """
        Right so this is the actual strategy. At least this the most basic form of it, planning on interating up
        :param portfolio: The portfolio of data in the getOneSector return format
        :param params: Parameters to the strategy as a dictionary I think
        """
        self.data = portfolio
        self.format = "%Y-%m-%d"
        # parameters
        self.coint_check_length = params['lookback']
        self.use_check_for = params['lookforward']
        # Start Date of the data entered
        self.startPortfolio = portfolio[list(portfolio.keys())[0]].index[0]
        self.endPortfolio = portfolio[list(portfolio.keys())[0]].index[-1]


        # Work out net price of opening position here.

    def generateSignals(self):
        """
        This is the big dog, the core of the strategy. This finds cointegration between pairs in the given portfolio
        over the lookback paramter given. And then generate signals on those pairs it deems cointegrated using the
        paramters calculated over the lookback period to avoid look ahead bias
        :return: Dictionary with key as the pair name and value as a PairOutput class which should contain all the info
        needed by the backtester
        """
        start_coint_check = self.startPortfolio
        end_coint_check = start_future = add_time(start_coint_check, month=self.coint_check_length)
        end_future = add_time(start_future, month=self.use_check_for)
        # Now lets find the cointegrated pairs I'm going to work on for the next 3 months.
        cointAnalysis = CointAnalysis(self.data)
        cointPairs = cointAnalysis.checkPortfolioForCoint(critValue=0.005, fromDate=start_coint_check, toDate=end_coint_check)
        dict_of_pairs = {}
        for index, pair in cointPairs.iterrows():
            pair_name = str(pair.symbol1) + str(pair.symbol2)
            pastdata1 = self.data[pair.symbol1]["adjusted_close"][start_coint_check:end_coint_check]
            pastdata2 = self.data[pair.symbol2]["adjusted_close"][start_coint_check:end_coint_check]
            futuredata1 = self.data[pair.symbol1]["adjusted_close"][start_future:end_future]
            futuredata2 = self.data[pair.symbol2]["adjusted_close"][start_future:end_future]
            past_z_score, past_mean, past_std, current_OLS = cointAnalysis.calc_z_score([pair.symbol1, pair.symbol2], fromDate=start_coint_check, toDate=end_coint_check)
            if current_OLS.params[1] < 0:
                # This should probably be introduced into the cointegrated pair function
                ## This is saying if B is negative don't include it as it means you won't be market neutral, you will be longing/shorting both.
                continue
            # plt.plot(pastdata1 - OLS.params[1] * pastdata2 - OLS.params[0])
            # plt.plot(past_z_score)
            # plt.show()
            future_z_score, _, _, _ = cointAnalysis.calc_z_score([pair.symbol1, pair.symbol2], fromDate=start_future, toDate=end_future, mean=past_mean, std=past_std, OLS=current_OLS)
            # p represents longing or shorting the spread. You short the spread if higher than expected, long the spread if lower than expected
            p = np.zeros(len(future_z_score))
            close = np.zeros(len(future_z_score))
            # cointAnalysis.plot_pair(pair, fromDate=start_coint_check, toDate=end_future)
            # These plots are really handy sometimes so gonna keep them in apologies for the mess
            # fig, axs = plt.subplots(3)
            # all_z_score = np.append(past_z_score, future_z_score)
            # axs[0].plot(past_z_score)
            # axs[0].axhline(0, linestyle='--')
            # axs[0].axhline(1, linestyle='--')
            # axs[0].axhline(-1, linestyle='--')
            # axs[0].xaxis.set_major_locator(plt.MaxNLocator(10))
            # axs[1].plot(future_z_score)
            # axs[1].axhline(0, linestyle='--')
            # axs[1].axhline(1, linestyle='--')
            # axs[1].axhline(-1, linestyle='--')
            # axs[1].xaxis.set_major_locator(plt.MaxNLocator(10))
            # axs[2].plot(all_z_score)
            # axs[2].axhline(0, linestyle='--')
            # axs[2].axhline(1, linestyle='--')
            # axs[2].axhline(-1, linestyle='--')
            # axs[2].xaxis.set_major_locator(plt.MaxNLocator(10))
            for i in range(0, len(future_z_score)):
                # This generates the signals for the required period, p represents opening positons (1, means open long, -1 open short)
                # close represents closing positions (1 means close long, -1 means close short)
                if future_z_score[i] >= 1:
                    p[i] = -1
                if future_z_score[i] < 0:
                    close[i] = -1
                if future_z_score[i] <= -1:
                    p[i] = 1
                if future_z_score[i] > 0:
                    close[i] = 1
            df = pd.DataFrame({'close1': futuredata1, 'close2': futuredata2, 'z_score': future_z_score, 'p': p, 'close': close})
            # plt.figure()
            # plt.scatter(futuredata1, futuredata2)
            current_pair = PairOutput()
            current_pair.OLS = current_OLS
            current_pair.mean = past_mean
            current_pair.std = past_std
            current_pair.df = df
            dict_of_pairs[pair_name] = current_pair
        return dict_of_pairs




if __name__ == "__main__":
    x = DataManager()
    # This code will just do it for one sector
    # x.data = x.getOneSector(sector="Energy", fromDate="2015-01-01", toDate="2016-09-21")
    x.getOneSector(sector="Energy", fromDate="2014-01-01", toDate="2018-01-01")
    #x.calcReturns()

    strat = CointStrategy(x.data, params={'lookback': 24, 'lookforward': 12})
    strat.generateSignals()
