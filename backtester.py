from DataManager import DataManager
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson, zscore
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
# from statsmodels.regression.linear_model import sm.OLS
import statsmodels.api as sm
from cointStrategy import CointStrategy
from utils import calc_diff


class Backtester:
    def __init__(self, strat, portfolio):
        """
        This is the function that actually backtests the strategy to see how it would have done over a given period
        :param strat: This is the strategy to backtested
        :param portfolio: The portfolio to run the backtest on
        """
        self.strat = strat
        self.portfolio = portfolio
        self.format = "%Y-%m-%d"
        self.startPortfolio = portfolio[list(portfolio.keys())[0]].index[0]
        self.endPortfolio = portfolio[list(portfolio.keys())[0]].index[-1]

        # This strat output should be a dict with key the name of pair of pairdfs where pairdfs are dataframes with
        # columns={Date, open1, open2, close1, close2, p} where p is the signals
        self.params = {'lookback': 24, 'lookforward': 12}
        self.strat_output = self.strat(self.portfolio, params=self.params).generateSignals()
        self.beta = None
        self.open_pos = None
        self.entry_price = None
        self.entry_date = None
        self.close_price = None
        self.returns_dict = {}
        self.trans_dict = {}

    def reset_vars(self):
        """
        This resets all variables, usually used on a close position
        :return:
        """
        self.entry_price = None
        self.entry_date = None
        self.close_price = None
        self.open_pos = None

    def openPos(self, date, row):
        # Direction means long or short
        """
        This function opens a positon, long or short.
        :param date: the date which the positon opens
        :param row: the row of data it is opening on. This represents the pair data, so the price of each etc
        :return: None but appends to transaction dataframe that this transaction has happened
        """
        direction = row.p
        self.open_pos = direction
        self.entry_date = date
        self.entry_price = (row.close1 - self.beta*row.close2)
        self.trans_df = self.trans_df.append(
            {'date': date, 'direction': direction, 'close': 0, 'price1': row.close1, 'price2': row.close2}, ignore_index=True)

    def closePos(self, date, row):
        """
        This function closes a positon and calculates the returns for the transaction
        :param date: the date which the positon opens
        :param row: the row of data it is opening on. This represents the pair data, so the price of each etc
        :return: None but appends to returns df and trans df
        """
        direction = row.p
        days_held = calc_diff(self.entry_date, date)
        self.close_price = (row.close1 - self.beta*row.close2)
        sign = 1
        if self.entry_price < 0 and self.close_price < 0:
            sign = -1
        returns = (self.close_price/self.entry_price - 1) * row.close * sign
        self.trans_df = self.trans_df.append(
            {'date': date, 'direction': row.close, 'close': 1, 'price1': row.close1, 'price2': row.close2}, ignore_index=True)
        self.return_df = self.return_df.append({'date': date, 'returns': returns, 'days_held': days_held}, ignore_index=True)
        # reset Variables
        self.reset_vars()



    def backtest(self):
        # TODO: At some point I'll likely need to swap these loops round so that it loops through date and then each pair
        # instead of all dates of one pair
        """
        This actually runs the backtest. Its worth looking at what strat_output looks like
        :return:
        """

        for pair, output_class in self.strat_output.items():
            self.beta = output_class.OLS.params[1]
            self.trans_df = pd.DataFrame(
                columns=['date', 'direction', 'close', 'price1', 'price2'])
            self.return_df = pd.DataFrame(
                columns=['date', 'returns', 'days_held'])
            for date, row in output_class.df.iterrows():
                # Open up a returns and transaction df

                # If no position is opened open position
                if self.open_pos is None:
                    if row['p'] == 1 or row['p'] == -1:
                        self.openPos(date, row)
                # if a long position is open and strat says to close then close it
                elif self.open_pos == 1:
                    if row['close'] == 1:
                        self.closePos(date, row)
                # if a short position is open and strat says to close then close it
                elif self.open_pos == -1:
                    if row['close'] == -1:
                        self.closePos(date, row)


            if self.open_pos==1 or self.open_pos==-1:
                # TODO: Need some way of incorporating this into returns
                print("Position is still open do something about this")
                self.reset_vars()

            self.returns_dict[pair] = self.return_df
            self.trans_dict[pair] = self.trans_df




if __name__ == "__main__":
    x = DataManager()
    # This code will just do it for one sector
    # x.data = x.getOneSector(sector="Energy", fromDate="2015-01-01", toDate="2016-09-21")
    x.getOneSector(sector="Energy", fromDate="2014-01-01", toDate="2018-01-01")
    # x.calcReturns()

    strat = CointStrategy
    bt = Backtester(strat, x.data)
    bt.backtest()
    # This is just here so i can set a breakpoint and dig into the data.
    print("HI")

