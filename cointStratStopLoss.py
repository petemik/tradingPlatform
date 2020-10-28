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
import time
from cointStrategy import PairOutput

"""
Builds on: cointStrategy
Addition: Adds an extra step to the buy condition that requires it reverts direciton before buying. Also adds a stop loss.
Next Step: Add some sort of rolling window as at the moment it only works for one 4 month period
"""

class CointStrategyStopLoss:

    def __set_parameters(self, params):
        try:
            self.coint_check_length = params['lookback']
        except KeyError:
            self.coint_check_length = 10
        try:
            self.use_check_for = params['lookforward']
        except KeyError:
            self.use_check_for = 4
        try:
            self.critValue = params['critValue']
        except KeyError:
            self.critValue = 0.05
        try:
            # The std deviation of which to buy
            self.buy_cutoff = params['buy_cutoff']
        except KeyError:
            self.buy_cutoff = 1
        try:
            # The std deviation of which to buy
            self.sell_cutoff = params['sell_cutoff']
        except KeyError:
            self.sell_cutoff = 0


    def __init__(self, portfolio, params):
        """
        Right so this is the actual strategy. At least this the most basic form of it, planning on interating up
        :param portfolio: The portfolio of data in the getOneSector return format
        :param params: Parameters to the strategy as a dictionary I think
        """
        # Parameters
        self.coint_check_length = None
        self.use_check_for = None
        self.critValue = None
        self.buy_cutoff = None
        self.sell_cutoff = None
        self.__set_parameters(params)

        self.data = portfolio
        self.format = "%Y-%m-%d"
        # Start Date of the data entered
        self.startPortfolio = portfolio[list(portfolio.keys())[0]].index[0]
        self.endPortfolio = portfolio[list(portfolio.keys())[0]].index[-1]
        self.CA = CointAnalysis(self.data)
        self.start_coint_check = self.startPortfolio
        self.end_coint_check = self.start_future = add_time(self.start_coint_check, month=self.coint_check_length)
        self.end_future = add_time(self.start_future, month=self.use_check_for)
        # Gathers all the cointegrated pairs over the last n months.
        self.cointPairs = self.CA.checkPortfolioForCoint(critValue=self.critValue, fromDate=self.start_coint_check,
                                                               toDate=self.end_coint_check)



    def generateSignals(self):
        """
        This is the big dog, the core of the strategy. This finds cointegration between pairs in the given portfolio
        over the lookback paramter given. And then generate signals on those pairs it deems cointegrated using the
        paramters calculated over the lookback period to avoid look ahead bias
        :return: Dictionary with key as the pair name and value as a PairOutput class which should contain all the info
        needed by the backtester
        """
        start_time =time.time()
        # Now lets find the cointegrated pairs I'm going to work on for the next 3 months.
        dict_of_pairs = {}
        for index, pair in self.cointPairs.iterrows():
            pair_name = str(pair.symbol1) + ' ' + str(pair.symbol2)
            pastdata1 = self.data[pair.symbol1]["adjusted_close"][self.start_coint_check:self.end_coint_check]
            pastdata2 = self.data[pair.symbol2]["adjusted_close"][self.start_coint_check:self.end_coint_check]
            futuredata1 = self.data[pair.symbol1]["adjusted_close"][self.start_future:self.end_future]
            futuredata2 = self.data[pair.symbol2]["adjusted_close"][self.start_future:self.end_future]
            past_z_score, past_mean, past_std, current_OLS = self.CA.calc_z_score([pair.symbol1, pair.symbol2],
                                                                                  fromDate=self.start_coint_check, toDate=self.end_coint_check)
            # plt.plot(pastdata1 - OLS.params[1] * pastdata2 - OLS.params[0])
            # plt.plot(past_z_score)
            # plt.show()
            future_z_score, _, _, _ = self.CA.calc_z_score([pair.symbol1, pair.symbol2], fromDate=self.start_future,
                                                           toDate=self.end_future, mean=past_mean, std=past_std, OLS=current_OLS)
            # p represents longing or shorting the spread. You short the spread if higher than expected, long the spread if lower than expected
            p = np.zeros(len(future_z_score))
            close = np.zeros(len(future_z_score))
            set_signal = 0
            store_set_signal = np.zeros(len(future_z_score))
            store_position_open = np.zeros(len(future_z_score))
            # 1 means long open, -1 short open, 0 no position
            position_open = 0
            should_trade = True
            for i in range(0, len(future_z_score)):
                # This generates the signals for the required period, p represents opening positons (1, means open long, -1 open short)
                # close represents closing positions (1 means close long, -1 means close short)
                if should_trade is True:
                    if future_z_score[i] >= self.buy_cutoff and position_open == 0:
                        # Prepare to short if it drops below cut off
                        set_signal = -1
                    if self.sell_cutoff < future_z_score[i] <= 0.75*self.buy_cutoff and set_signal == -1 and position_open == 0:
                        # Short now that is has gone below cut off
                        p[i] = -1
                        position_open = -1
                        set_signal = 0
                    if future_z_score[i] < self.sell_cutoff and position_open == -1:
                        # If it drops below the sell_cut_off criteria then close the short position and position is close
                        close[i] = -1
                        position_open = 0
                    if future_z_score[i] > 1.25*self.buy_cutoff and position_open == -1:
                        # This is the stop loss
                        close[i] = -1
                        position_open = 0
                        should_trade = False
                    if set_signal == -1 and self.sell_cutoff > future_z_score[i]:
                        set_signal = 0

                    if future_z_score[i] <= -self.buy_cutoff and position_open == 0:
                        # Prepare to buy
                        set_signal = 1
                    if 0.75*(-self.buy_cutoff) <= future_z_score[i] < self.sell_cutoff and set_signal == 1 and position_open==0:
                        # Buy
                        p[i] = 1
                        position_open = 1
                        set_signal = 0
                    if future_z_score[i] > -self.sell_cutoff and position_open == 1:
                        # Sell condition
                        close[i] = 1
                        position_open = 0
                    if future_z_score[i] < -1.25*self.buy_cutoff and position_open == 1:
                        # Stop loss
                        close[i] = 1
                        position_open = 0
                        should_trade = False
                    if set_signal == 1 and self.sell_cutoff < future_z_score[i]:
                        # Thia just resets the signal if it shoots out the other end without ever getting bough, rapid changes.
                        set_signal = 0
                    store_set_signal[i] = set_signal
                    store_position_open[i] = int(position_open)
            df = pd.DataFrame({'close1': futuredata1, 'close2': futuredata2, 'z_score': future_z_score, 'p': p, 'close': close, 'set_signal': store_set_signal, 'position_open': store_position_open})
            # plt.figure()
            # plt.scatter(futuredata1, futuredata2)
            current_pair = PairOutput()
            current_pair.OLS = current_OLS
            current_pair.mean = past_mean
            current_pair.std = past_std
            current_pair.df = df
            dict_of_pairs[pair_name] = current_pair
        print("generateSignals took {} seconds".format(time.time() - start_time))
        return dict_of_pairs

    def naughty_plot(self):

        for index, pair in self.cointPairs.iterrows():
            pastdata1 = self.data[pair.symbol1]["adjusted_close"][self.start_coint_check:self.end_coint_check]
            pastdata2 = self.data[pair.symbol2]["adjusted_close"][self.start_coint_check:self.end_coint_check]
            futuredata1 = self.data[pair.symbol1]["adjusted_close"][self.start_future:self.end_future]
            futuredata2 = self.data[pair.symbol2]["adjusted_close"][self.start_future:self.end_future]
            past_z_score, past_mean, past_std, current_OLS = self.CA.calc_z_score([pair.symbol1, pair.symbol2],
                                                                                  fromDate=self.start_coint_check,
                                                                                  toDate=self.end_coint_check)
            # self.CA.plot_pair(pair, fromDate=self.start_coint_check, toDate=self.end_future)
            future_z_score, _, _, _ = self.CA.calc_z_score([pair.symbol1, pair.symbol2], fromDate=self.start_future,
                                                           toDate=self.end_future, mean=past_mean, std=past_std,
                                                           OLS=current_OLS)
            # These plots are really handy sometimes so gonna keep them in apologies for the mess
            fig, axs = plt.subplots(3)
            all_z_score = np.append(past_z_score, future_z_score)
            axs[0].plot(past_z_score)
            axs[0].axhline(0, linestyle='--')
            axs[0].axhline(1, linestyle='--')
            axs[0].axhline(-1, linestyle='--')
            axs[0].axhline(2, linestyle='--')
            axs[0].axhline(-2, linestyle='--')
            axs[0].xaxis.set_major_locator(plt.MaxNLocator(10))
            axs[1].plot(future_z_score)
            axs[1].axhline(0, linestyle='--')
            axs[1].axhline(1, linestyle='--')
            axs[1].axhline(-1, linestyle='--')
            axs[1].axhline(2, linestyle='--')
            axs[1].axhline(-2, linestyle='--')
            axs[1].xaxis.set_major_locator(plt.MaxNLocator(10))
            axs[2].plot(all_z_score)
            axs[2].axhline(0, linestyle='--')
            axs[2].axhline(1, linestyle='--')
            axs[2].axhline(-1, linestyle='--')
            axs[2].axhline(2, linestyle='--')
            axs[2].axhline(-2, linestyle='--')
            axs[2].xaxis.set_major_locator(plt.MaxNLocator(10))
            fig.suptitle('Plot of {} and {} with pvalue {}'.format(pair.symbol1, pair.symbol2, pair.pvalue), fontsize=16)
            axs[0].set_title("The Z-score for the lookback period")
            axs[1].set_title("The Z-score for the upcoming period")
            axs[2].set_title("The Z-score for the entire period")



if __name__ == "__main__":
    x = DataManager()
    # This code will just do it for one sector
    # x.data = x.getOneSector(sector="Energy", fromDate="2015-01-01", toDate="2016-09-21")
    x.getOneSector(sector="Financials", fromDate="2014-01-01", toDate="2018-01-01")
    #x.calcReturns()

    #strat = CointStrategy(x.data, params={'lookback': 24, 'lookforward': 12})
    # strat.generateSignals()
