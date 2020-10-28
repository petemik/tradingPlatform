from DataManager import DataManager
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import statsmodels.api as sm
from scipy.stats import poisson, zscore
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
# from statsmodels.regression.linear_model import sm.OLS
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import time
from utils import add_time, subtract_time, calc_diff
import winsound


class CointAnalysis():
    def __init__(self, portfolio, onReturns=False):
        """
        This class is used to analyse the portfolio for cointegration in various ways.
        :param portfolio: Data in the dictionary format explained in datamanager
        :param onReturns: Initially I played around with calculating cointegration on returns vs adjusted close,
        determined adjusted_close is much better but this boolean changes which one it finds cointegration on
        """
        self.portfolio = portfolio
        self.cointStocks = None
        self.format = "%Y-%m-%d"
        self.onReturns = onReturns
        if not self.onReturns:
            self.analysisOn = 'adjusted_close'
        else:
            self.analysisOn = 'logReturns'

    def checkPortfolioForCoint(self, critValue=0.01, fromDate="2015-01-01", toDate="2020-09-21", calcMean=False,
                               usead=0):
        """
        This is one of the most useful function in this class
        :param critValue: The critical value of which to judge things are cointegrated, equivalent to the p value cut off
        :param fromDate: The date to start the check of cointegration test from
        :param toDate: The date to end the check of cointegration test
        :param calcMean: Sometimes it is handy to return the mean of the cointegrated data, this gives a boolean for doing that
        :param usead: I tested other cointegration methods and this allows you to adjust which one to use
        :return: The dataframe of (symbol1, symbol2, pvalue), so the 2 cointegrated stocks and the certainty of which they are cointegrated
        """
        start_time = time.time()
        num_stocks = len(self.portfolio)
        keys = list(self.portfolio.keys())
        df = pd.DataFrame(columns=('symbol1', 'symbol2', 'pvalue'))
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                data1 = self.portfolio[keys[i]][self.analysisOn][fromDate:toDate]
                data2 = self.portfolio[keys[j]][self.analysisOn][fromDate:toDate]
                model = sm.OLS(data1, sm.add_constant(data2))
                results = model.fit()
                spread = data1 - results.params[1] * data2 - results.params[0]
                if results.params[1] < 0:
                    continue
                try:
                    if not usead:
                        result = coint(data1, data2)
                    else:
                        result = adfuller(spread)
                    # else:
                    #     df = pd.DataFrame({'data1': data1, 'data2': data2})
                    #     result = coint_johansen(df, 0, 1)
                    #     print("Critical values(90%, 95%, 99%) of max_eig_stat\n", result.cvm, '\n')
                    #     print("Critical values(90%, 95%, 99%) of trace_stat\n", result.cvt, '\n')

                except:
                    print("Cannot calculate coint for {}, {}".format(keys[i], keys[j]))
                    continue
                pvalue = result[1]
                if pvalue < critValue:
                    mean = np.mean(spread)
                    std = np.std(spread)
                    df = df.append({'symbol1': keys[i], 'symbol2': keys[j], 'pvalue': pvalue, 'mean': mean, 'std': std},
                                   ignore_index=True)
                    # At some point add sm.OLS HERE

        df = df.sort_values(by='pvalue', ignore_index=True)
        self.cointStocks = df
        print("checkPortfolioForCoint took {} seconds".format(time.time()-start_time))
        if calcMean == False:
            return df

    def return_pvalues_portfolio(self, fromDate="2015-01-01", toDate="2020-09-21"):
        """
        This is one of the most useful function in this class
        :param critValue: The critical value of which to judge things are cointegrated, equivalent to the p value cut off
        :param fromDate: The date to start the check of cointegration test from
        :param toDate: The date to end the check of cointegration test
        :param calcMean: Sometimes it is handy to return the mean of the cointegrated data, this gives a boolean for doing that
        :param usead: I tested other cointegration methods and this allows you to adjust which one to use
        :return: The dataframe of (symbol1, symbol2, pvalue), so the 2 cointegrated stocks and the certainty of which they are cointegrated
        """
        start_time = time.time()
        num_stocks = len(self.portfolio)
        keys = list(self.portfolio.keys())
        df = pd.DataFrame(columns=('symbol1', 'symbol2', toDate))
        for i in range(num_stocks):
            for j in range(i + 1, num_stocks):
                data1 = self.portfolio[keys[i]][self.analysisOn][fromDate:toDate]
                data2 = self.portfolio[keys[j]][self.analysisOn][fromDate:toDate]
                try:
                    result = coint(data1, data2)
                except:
                    print("Cannot calculate coint for {}, {}".format(keys[i], keys[j]))
                    continue
                pvalue = result[1]
                df = df.append({'symbol1': keys[i], 'symbol2': keys[j], toDate: pvalue},
                                   ignore_index=True)

        print("return_pvalues_portfolio took {} seconds".format(time.time()-start_time))
        return df

    def check_significance(self, num_pairs, critValue, cut_off=0.10):
        """
        This is mostly a playing about function, this tests to see if the number of calculated cointegration pairs is
        more than you expected from random deviation to see if the results are significant
        i.e. If you calculate 12 pairs when you tests 100 at a 0.1 critvalue is that significant?
        This is used to adjust for multiple comparison bias
        :param num_pairs: The number of calculated pairs
        :param critValue: The critical value you used
        :param cut_off: The cut off to determine whether or not the results are significant
        :return:
        """
        num_stocks = len(self.portfolio)
        num_comparison = (num_stocks) * (num_stocks - 1) / 2
        x_val = round(critValue * num_comparison)
        fish = 1 - poisson.cdf(k=num_pairs - 1, mu=x_val)
        if fish < cut_off:
            print("These results seem significant: \n Expected: {} | Pairs: {} | p: {}".format(x_val, num_pairs, fish))
            return 1
        else:
            print(
                "These results may be insignificant:\n Expected: {} | Pairs: {} | p: {}".format(x_val, num_pairs, fish))
            return 0

    def calc_z_score(self, pair, mean=None, std=None, OLS=None, fromDate="2015-01-01", toDate="2020-09-21"):
        """
        This calculates the zscore of a pair (i.e. the spread)
        :param pair: The pair you want to calculate the z_score for as a tuple or list [synbol1, symbol2]
        :param mean: This is used if you want to assert a mean to be used,
        this is necessary to avoid lookahead bias sometimes
        :param std: This is used if you want to assert a std to be used,
        this is necessary to avoid lookahead bias sometimes
        :param OLS: This is used if you want to assert a Beta value to be used,
        this is necessary to avoid lookahead bias sometimes
        :param fromDate: The date to calculate the z_score from
        :param toDate:The date to calculate the z_score to
        :return: The z_score and the mean, std, and OLS used (will explain why this is useful later in the code)
        """
        symbol1 = pair[0]
        symbol2 = pair[1]
        data1 = self.portfolio[symbol1][self.analysisOn][fromDate:toDate]
        data2 = self.portfolio[symbol2][self.analysisOn][fromDate:toDate]
        if OLS is None:
            model = sm.OLS(data1, sm.add_constant(data2))
            results = model.fit()
        else:
            results = OLS
        # need to use same OLS
        spread = data1 - results.params[1] * data2 - results.params[0]

        if mean is None: mean = np.mean(spread)
        future_mean = np.mean(spread)
        if std is None: std = np.std(spread)
        z_score = (spread - mean) / std
        return (z_score, mean, std, results)

    def plot_pair(self, pair, fromDate="2015-01-01", toDate="2020-09-21"):
        """
        Plots the pair seperately and the z_score of the spread
        :param pair: The pair to be used
        :param fromDate: Date from
        :param toDate: Date to
        :return: Empty as is a plot
        """
        symbol1 = pair[0]
        symbol2 = pair[1]
        data1 = self.portfolio[symbol1][self.analysisOn][fromDate:toDate]
        data2 = self.portfolio[symbol2][self.analysisOn][fromDate:toDate]
        # ratio = data1 / data2
        model = sm.OLS(data1, sm.add_constant(data2))
        results = model.fit()
        spread = data1 - results.params[1] * data2 - results.params[0]
        z_score = zscore(spread)
        mean_z = np.mean(z_score)
        std_z = np.std(z_score)
        fig, axs = plt.subplots(2)
        axs[0].plot(data1)
        axs[0].plot(data2)
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(10))
        axs[1].plot(z_score)
        axs[1].axhline(mean_z, linestyle='--')
        axs[1].axhline(mean_z + std_z, linestyle='--')
        axs[1].axhline(mean_z - std_z, linestyle='--')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.title("Showing the pair comparison of {} and {}".format(symbol1, symbol2))

    def rolling_z_score(self, pair, fromDate="2015-01-01", toDate="2020-09-21", window=30):
        """
        This tested out the concept of using a rolling z_score, will likely be introduced later but don't really use atm
        :param pair: The pair to calculate for
        :param fromDate: date from
        :param toDate: date to
        :param window: window to be used
        :return: none but plots the rolling z_score, like I said might be used in the future
        """
        symbol1 = pair[0]
        symbol2 = pair[1]
        data1 = self.portfolio[symbol1]['adjusted_close'][fromDate:toDate]
        data2 = self.portfolio[symbol2]['adjusted_close'][fromDate:toDate]
        model = sm.OLS(data1, sm.add_constant(data2))
        results = model.fit()
        spread = data1 - results.params[1] * data2 - results.params[0]
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()

        rolling_z_dcore = (spread - spread_mean) / spread_std
        mean_z = np.mean(rolling_z_dcore)
        std_z = np.std(rolling_z_dcore)
        plt.figure()
        plt.plot(rolling_z_dcore)
        plt.axhline(mean_z, linestyle='--')
        plt.axhline(mean_z + std_z, linestyle='--')
        plt.axhline(mean_z - std_z, linestyle='--')
        ax = plt.axes()
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.show()

    def plot_pairs(self, pairs, n_plots=20, fromDate="2015-01-01", toDate="2020-09-21"):
        """
        This uses the plot_pair function but can plot a bunch of pairs, this is so you can quickly visualise the list of cointegrated pairs
        :param pairs: The pairs you want to plot as returned from the checkPortfolio for coint function
        :param n_plots The number of pairs you want to see as sometimes opens like 20
        :param fromDate: date from
        :param toDate: date to
        :return: None just makes a bunch of plots
        """
        n_pairs = len(pairs)
        n_plots = min(n_plots, n_pairs)
        for i in range(0, n_plots):
            pair = (pairs['symbol1'][i], pairs['symbol2'][i])
            self.plot_pair(pair, fromDate=fromDate, toDate=toDate)

    def findCointPairs(self, fromDate="2016-01-01", toDate=None, critValue=0.01, minMonthsBack=9):
        """
        This was used to add a bit more complexity to the check portfolio for coint function, as i found which
        symbols were cointegrated very frequently changed (which defeats the point of cointegration). yet I could
        also see some pairs over certain time frames were very clearly cointegrated. It was also saying some imo very
        clearly not cointegrated pairs were cointegrated. This was supposed to be a bit more thorough by checking if
        the pair is cointegrated over multiple different time frames (Or at least over half of the different time
        frames tested)
        :param fromDate: date from
        :param toDate: date to
        :param critValue: critical value, if none just calculate for the current month
        :param minMonthsBack: the intial number of months aback to check cointegrations, so will beck if cointegration over minMonthsBack, minMonths + 1, +2, +3,+4
        :return: The cointegrated pairs more thoroughly calculated
        """
        fromDateDatetime = datetime.strptime(fromDate, self.format)
        if toDate == None:
            toDate = (fromDateDatetime + relativedelta(months=1)).strftime(self.format)
        toDateDatetime = datetime.strptime(toDate, self.format)
        numMonths = (toDateDatetime.year - fromDateDatetime.year) * 12 + (toDateDatetime.month - fromDateDatetime.month)
        theCointegratedLot = []
        for i in range(0, numMonths):
            currentToDate = (fromDateDatetime + relativedelta(months=i))
            currentFromDate = currentToDate - relativedelta(months=minMonthsBack)

            pairsdf = self.checkPortfolioForCoint(critValue=critValue,
                                                  fromDate=currentFromDate.strftime(
                                                      self.format), toDate=currentToDate.strftime(
                    self.format))
            for j in range(1, 5):
                pairsdf = pairsdf.append(self.checkPortfolioForCoint(critValue=critValue,
                                                                     fromDate=(currentFromDate - relativedelta(
                                                                         months=j)).strftime(
                                                                         self.format),
                                                                     toDate=currentToDate.strftime(self.format)),
                                         ignore_index=True)
            grouped = pairsdf.groupby(['symbol1', 'symbol2'], group_keys=False).size().reset_index(name='counts')
            grouped = grouped[grouped.counts > 3]
            theCointegratedLot.append(grouped)
            print("Finished month n: {}".format(i))
            print("Results: \n {}".format(grouped))
            # Do calculation on this to calculate the average pvalue only if it occurs 4 or more times in the set (This
            # will likely need another function to do it. Simplify(pairddf) or something) Store these values. probably
            # seperate this bit out into one function. Have just the end date, calc and then return definite pairs df.
            # And then interate through the months and return all the dataframes in a dict or something maybe? Maybe in a
            # new bigger dataframe. Once this is done can see how well this function picks out cointegrated stocks over a series of time.
            # In future could potentially run again on this dataframe to see if there any really persistently cointegrated stocks
        return theCointegratedLot

    def seeHowPchanges(self, pair, fromDate="2016-01-01", toDate="2020-09-21", cumulative=False, lookback=12):
        """
        This visualises the phenomena i was referring to in the findCointPairs function where the p value that
        represents likelihood of cointegrations was varying massively over different time frames. So this plots how
        the p value changes over time to see if there is a patter in when it breaks
        :param pair: The pair to check the p value over time for
        :param fromDate: Date from
        :param toDate: Date to
        :param cumulative: Hmm difficult to explain, see code
        :param lookback: The period of which to lookback to calculate p over. So for example using the default parameters
        we will check the pvalue (caclculated for the past year from the current Date) every week from 2016 to 2020
        :return: A useful dataframe for testing
        """
        fromDateDatetime = datetime.strptime(fromDate, self.format)
        toDateDatetime = datetime.strptime(toDate, self.format)
        numMonths = (toDateDatetime.year - fromDateDatetime.year) * 12 + (toDateDatetime.month - fromDateDatetime.month)
        numWeeks = round((toDateDatetime - fromDateDatetime).days / 7)
        # numDays =(toDateDatetime - fromDateDatetime).days
        fakedf = pd.DataFrame(columns=('dateFrom', 'dateTo', 'pvalue'))
        pvalues = []
        for i in range(0, numWeeks):
            currentToDate = fromDateDatetime + relativedelta(weeks=i)
            if cumulative == False:
                currentFromDate = currentToDate - relativedelta(months=lookback)
            else:
                currentFromDate = fromDateDatetime - relativedelta(months=lookback)
            data1 = self.portfolio[pair[0]][self.analysisOn][
                    currentFromDate.strftime(self.format):currentToDate.strftime(self.format)]
            data2 = self.portfolio[pair[1]][self.analysisOn][
                    currentFromDate.strftime(self.format):currentToDate.strftime(self.format)]
            result = coint(data1, data2)
            pvalue = result[1]
            fakedf = fakedf.append({"dateFrom": currentFromDate, "dateTo": currentToDate, "pvalue": pvalue},
                                   ignore_index=True)
        return fakedf

    def mean_reversion_rolling(self, pair, fromDate="2015-01-01", toDate="2018-01-01"):
        """
        This looked at using a more basic concept than cointegration, just looking for mean reversion but decided against it
        :param pair: Pair to look at mean reversion for
        :param fromDate:  date from
        :param toDate:  date to
        :return:
        """
        symbol1 = pair[0]
        symbol2 = pair[1]
        data1 = self.portfolio[symbol1][self.analysisOn][fromDate:toDate]
        data2 = self.portfolio[symbol2][self.analysisOn][fromDate:toDate]
        model = sm.OLS(data1, sm.add_constant(data2))
        results = model.fit()
        spread = data1 - results.params[1] * data2 - results.params[0]
        z_score = zscore(spread)
        mean_z = np.mean(z_score)
        spread_mean = pd.Series(z_score).rolling(window=30).mean()
        std_z = np.std(z_score)
        plt.plot(z_score)
        plt.plot(spread_mean)
        plt.show()

    def plot_various_p_and_z_score(self, pair, fromDate="2015-01-01", toDate="2018-01-01"):
        """
        Pfft this is very much a testing an idea out function. This looks at how pvalue changes over time (uisng various windows) and plots it against the z_score
        Honestly is probably easier just to run it and see what happens
        :param pair: pair to check for
        :param fromDate: date from
        :param toDate: date tp
        :return: A bunch of plots
        """
        lookback6 = self.seeHowPchanges(pair=pair, fromDate=fromDate, toDate=toDate, lookback=6)
        lookback9 = self.seeHowPchanges(pair=pair, fromDate=fromDate, toDate=toDate, lookback=9)
        lookback12 = self.seeHowPchanges(pair=pair, fromDate=fromDate, toDate=toDate, lookback=12)
        lookback15 = self.seeHowPchanges(pair=pair, fromDate=fromDate, toDate=toDate, lookback=15)

        symbol1 = pair[0]
        symbol2 = pair[1]
        data1 = self.portfolio[symbol1][self.analysisOn][fromDate:toDate]
        data2 = self.portfolio[symbol2][self.analysisOn][fromDate:toDate]
        # ratio = data1 / data2
        ratio = data1 / data2
        z_score = zscore(ratio)
        mean_z = np.mean(z_score)
        std_z = np.std(z_score)
        fig, axs = plt.subplots(2)
        axs[0].plot(lookback6.dateTo, lookback6.pvalue)
        axs[0].plot(lookback6.dateTo, lookback9.pvalue)
        axs[0].plot(lookback6.dateTo, lookback12.pvalue)
        axs[0].plot(lookback6.dateTo, lookback15.pvalue)
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(15))
        axs[0].legend(['6 month', "9 month", "12 month", "15 month"])
        axs[1].plot(data1.index, z_score)
        axs[1].axhline(mean_z, linestyle='--')
        axs[1].axhline(mean_z + std_z, linestyle='--')
        axs[1].axhline(mean_z - std_z, linestyle='--')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(15))
        plt.show()

    def playing_with_rolling(self, pair, fromDate="2015-01-01", toDate="2018-01-01"):
        symbol1 = pair[0]
        symbol2 = pair[1]
        data1 = self.portfolio[symbol1][self.analysisOn][fromDate:toDate]
        data2 = self.portfolio[symbol2][self.analysisOn][fromDate:toDate]
        model = sm.OLS(data1, sm.add_constant(data2))
        window = 180
        model2 = RollingOLS(data1, sm.add_constant(data2), window=window)
        results = model.fit()
        results2 = model2.fit()
        # spread = data1 - results.params[1] * data2 - results.params[0]
        # spread_rolling = data1 - results2.params.adjusted_close * data2 - results2.params.const
        spread = data1 - results.params[1] * data2
        spread_rolling = data1 - results2.params.adjusted_close * data2
        spread_mean = pd.Series(spread_rolling).rolling(window=window).mean()
        spread_std = pd.Series(spread_rolling).rolling(window=window).std()
        fig, axs = plt.subplots(2)
        # plt.plot((spread - spread.mean())/spread.std())
        axs[0].plot((spread_rolling-spread_mean)/spread_std)
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(15))
        axs[1].plot(results2.params.adjusted_close['2013-03-15':])
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(15))
        # plt.plot(spread)
        # plt.plot(spread_rolling)
        plt.show()

    def generateCointegratedPairs(self, fromDate, toDate, cointOver=4, lookback=True):
        """
        This should return a dataframe with dates down the left and all pairs across the top, with a boolean of 0 if
        the pairs are not cointegrated over the time period looking backward, 1 if they are cointegrated over the time period
        :param fromDate: Date to start calculation of cointover
        :param toDate: date to end calculation of cointover
        :param cointOver:The number of months to test if something is cointegrated over. Too long and it likely breaks, too short and it doesn't have enough
        :return: The dataframe described above
        """
        if lookback:
            lookback_str = 'lookback'
        else:
            lookback_str = 'lookforward'
        monthsDiff = calc_diff(fromDate, toDate, type='months')
        array_of_end_dates = []

        for i in range(0, monthsDiff+1):
            array_of_end_dates.append(add_time(fromDate, month=i))
        columns = ['symbol1', 'symbol2'] + array_of_end_dates
        df = pd.DataFrame(columns=columns)
        if lookback:
            current_p_values = self.return_pvalues_portfolio(fromDate=subtract_time(fromDate, month=cointOver), toDate=fromDate)
        else:
            current_p_values = self.return_pvalues_portfolio(fromDate=fromDate,
                                                             toDate=add_time(fromDate, month=cointOver))
        df['symbol1'] = current_p_values['symbol1']
        df['symbol2'] = current_p_values['symbol2']
        df[fromDate] = current_p_values[add_time(fromDate, month=cointOver)]
        for current_date in array_of_end_dates:
            if current_date == fromDate:
                continue
            if lookback:
                current_p_values = self.return_pvalues_portfolio(fromDate=subtract_time(current_date, month=cointOver),
                                                             toDate=current_date)
            else:
                current_p_values = self.return_pvalues_portfolio(fromDate=current_date,
                                                                 toDate=add_time(current_date, month=cointOver))
            df[current_date] = current_p_values[add_time(current_date, month=cointOver)]
            print("{} data stored for {} {}".format(current_date, cointOver, lookback_str))
        return df

    def saveCointegratedPairs(self, fromDate, toDate, lookback=True, cointOverArray=np.arange(4, 14, 2)):
        if lookback:
            lookback_str = 'lookback'
        else:
            lookback_str = 'lookforward'
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        for i in cointOverArray:
            df = self.generateCointegratedPairs(fromDate=fromDate, toDate=toDate, cointOver=i, lookback=lookback)
            df.to_csv(modpath + "\Data\cointegratedPair\\" + lookback_str + "\\" + str(i) + "month" + ".csv")
            print("Successfully saved the {} month data".format(i))

    def saveBoth(self, fromDate, toDate, lookbackArray=None, lookforwardArray=None):
        if lookforwardArray is None:
            lookforwardArray = [4, 6, 8, 12]
        if lookbackArray is None:
            lookbackArray = np.arange(4, 14, 2)
        self.saveCointegratedPairs(fromDate=fromDate, toDate=toDate, lookback=False, cointOverArray=lookbackArray)
        self.saveCointegratedPairs(fromDate=fromDate, toDate=toDate, lookback=True, cointOverArray=lookforwardArray)




if __name__ == "__main__":
    # This is all a bit of a mess but I'll leave it in so people can see how some of these functions work

    x = DataManager()
    # # This code will just do it for one sector
    x.getOneSector(sector="Energy", fromDate="2011-01-01", toDate="2020-09-21")
    # x.calcReturns()
    # # Adjust this to avoid multiple comparison bias
    # critValue = 0.005
    cointAnalysis = CointAnalysis(x.data)
    # #cointAnalysis.playing_with_rolling(['APA', 'NBL'], fromDate="2010-01-01", toDate="2020-00-01", )
    # cointPairs1 = cointAnalysis.checkPortfolioForCoint(critValue=critValue, fromDate="2016-01-01", toDate="2016-04-01", usead=False)
    # #cointAnalysis.plot_pairs(cointPairs1, fromDate="2015-01-01", toDate="2016-06-01")
    # cointAnalysis.plot_pairs(cointPairs1, fromDate="2016-01-01", toDate="2016-04-01")
    # plt.show()
    # cointAnalysis.plot_pairs(cointPairs2, fromDate="2015-01-01", toDate="2016-01-01")
    # cointAnalysis.mean_reversion_rolling(pair=['APA', 'NBL'], fromDate="2015-01-01", toDate="2020-01-01")
    # cointAnalysis.plot_various_p_and_z_score(pair=['DVN', 'EOG'], fromDate="2014-01-01", toDate="2016-06-01")
    # cointAnalysis.plot_pair(pair=['APA', 'NBL'], fromDate="2015-01-01", toDate="2018-01-01")
    # plt.show()
    # cointAnalysis.generateCointegratedPairs('2014-01-01', '2014-04-01')
    cointAnalysis.saveCointegratedPairs(fromDate='2014-01-01', toDate='2017-01-01', lookback=False)

    duration = 500  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
