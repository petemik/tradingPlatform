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
from cointAnalysis import CointAnalysis
import re
from sklearn.metrics import confusion_matrix

class PredictCoint(CointAnalysis):
    def __lookback_str(self, lookback):
        if lookback:
            lookback_str = 'lookback'
        else:
            lookback_str = 'lookforward'

        return lookback_str
    def generateCointegratedPairs(self, fromDate, toDate, cointOver=4, lookback=True):
        """
        This should return a dataframe with dates down the left and all pairs across the top, with a boolean of 0 if
        the pairs are not cointegrated over the time period looking backward, 1 if they are cointegrated over the time period
        :param fromDate: Date to start calculation of cointover
        :param toDate: date to end calculation of cointover
        :param cointOver:The number of months to test if something is cointegrated over. Too long and it likely breaks, too short and it doesn't have enough
        :return: The dataframe described above
        """
        lookback_str = self.__lookback_str(lookback)
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
        if lookback:
            df[fromDate] = current_p_values[fromDate]
        else:
            df[fromDate] = current_p_values[add_time(fromDate, month=cointOver)]
        for current_date in array_of_end_dates:
            if current_date == fromDate:
                continue
            if lookback:
                current_p_values = self.return_pvalues_portfolio(fromDate=subtract_time(current_date, month=cointOver),
                                                             toDate=current_date)
                df[current_date] = current_p_values[current_date]
            else:
                current_p_values = self.return_pvalues_portfolio(fromDate=current_date,
                                                                 toDate=add_time(current_date, month=cointOver))
                df[current_date] = current_p_values[add_time(current_date, month=cointOver)]


            print("{} data stored for {} {}".format(current_date, cointOver, lookback_str))
        return df

    def saveCointegratedPairs(self, fromDate, toDate, lookback=True, cointOverArray=np.arange(4, 14, 2)):
        """
        This uses generate cointPairs over various different timeframes and saves them to a csv file
        :param fromDate: the usual
        :param toDate: the usual
        :param lookback: whether or not to lookforward or backward with your cointegration
        :param cointOverArray: the values of which to run the cointegration for.
        :return: None, just saves a bunch of csv files
        """
        lookback_str = self.__lookback_str(lookback)
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        for i in cointOverArray:
            df = self.generateCointegratedPairs(fromDate=fromDate, toDate=toDate, cointOver=i, lookback=lookback)
            df.to_csv(modpath + "\Data\cointegratedPair\\" + lookback_str + "\\" + str(i) + "month" + ".csv")
            print("Successfully saved the {} month data".format(i))

    def saveBoth(self, fromDate, toDate, lookbackArray=None, lookforwardArray=None):
        """
        Runs saveCointegratedPairs
        :param fromDate: usual
        :param toDate: usual
        :param lookbackArray: Windows to look back over for cointegration
        :param lookforwardArray: Windows to look forward over for cointegration
        :return: empty
        """
        if lookforwardArray is None:
            lookforwardArray = [4, 6, 8, 12]
        if lookbackArray is None:
            lookbackArray = np.arange(4, 14, 2)
        self.saveCointegratedPairs(fromDate=fromDate, toDate=toDate, lookback=False, cointOverArray=lookbackArray)
        self.saveCointegratedPairs(fromDate=fromDate, toDate=toDate, lookback=True, cointOverArray=lookforwardArray)

    def getDataSingle(self, window=4, lookback=True):
        """
        Retrieve one of the csvs and save it to a dataframe
        :param window: The cointegration window you want to retrieve
        :param lookback: Whether you want the lookback or lookforward window
        :return: A dataframe with the requested data
        """
        lookback_str = self.__lookback_str(lookback)
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        path_to_file = modpath + "\Data\cointegratedPair\\" + lookback_str + "\\" + str(window) + "month" + ".csv"
        df = pd.read_csv(path_to_file, index_col=0)
        return df

    def getDataGroup(self, lookback=True):
        """

        :param lookback: Whether to get all lookbacks or all lookforwards
        :return:
        """
        lookback_str = self.__lookback_str(lookback)
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        directory = modpath + "\Data\cointegratedPair\\" + lookback_str
        # path_to_file = modpath + "\Data\cointegratedPair\\" + lookback_str + "\\" + str(window) + "month" + ".csv"
        dict = {}
        for filename in os.listdir(directory):
            path_to_file = directory + "\\" + filename
            df = pd.read_csv(path_to_file, index_col=0)
            symbol = re.match(r"([0-9]+)([a-z]+)", filename, re.I).groups()[0]
            dict[symbol] = df
        return dict


    def reorgData(self):
        """
        I messed up the data I saved, not really in right format. So gonna reorganise it.
        :return:
        """
        lookback_str = "lookback"
        lookforward_str = "lookforward"
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        dir_lookback = modpath + "\Data\cointegratedPair\\" + lookback_str
        dir_lookforward = modpath + "\Data\cointegratedPair\\" + lookforward_str
        # Just use a random file to get the dates used
        path_to_file = modpath + "\Data\cointegratedPair\\" + lookback_str + "\\" + "4month" + ".csv"
        # lookback_dict = {}
        test_df = pd.read_csv(path_to_file, index_col=0)
        all_columns = test_df.columns
        just_dates = all_columns[2:]
        for i in range(0, len(just_dates)):
            new_df = test_df[test_df.columns[0:2]]
            for filename in os.listdir(dir_lookback):
                path_to_file = dir_lookback + "\\" + filename
                old_df = pd.read_csv(path_to_file, index_col=0)
                symbol = re.match(r"([0-9]+)([a-z]+)", filename, re.I).groups()[0]
                column_name = "lb" + str(symbol)
                new_df[column_name] = old_df[just_dates[i]]
            for filename in os.listdir(dir_lookforward):
                path_to_file = dir_lookforward + "\\" + filename
                old_df = pd.read_csv(path_to_file, index_col=0)
                symbol = re.match(r"([0-9]+)([a-z]+)", filename, re.I).groups()[0]
                column_name = "lf" + str(symbol)
                new_df[column_name] = old_df[just_dates[i]]
            new_df.to_csv(modpath + "\\Data\\cointegratedPair\\reorged\\" + str(just_dates[i]) + ".csv")
            del new_df

    def combine_data(self):
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        path_to_dir = modpath + "\\Data\\cointegratedPair\\reorged"
        first_file = os.listdir(path_to_dir)[0]
        df = pd.read_csv(path_to_dir + "\\" + first_file, index_col=0)
        for filename in os.listdir(path_to_dir):
            if not filename == first_file:
                current_df = pd.read_csv(path_to_dir + "\\" + filename, index_col=0)
                df = pd.concat([df, current_df], ignore_index=True, axis=0)
        return df


    def confusion_analysis(self, df):
        lookback = 15
        lookforward = 4
        new_df = self.convert_categories(df, critvalue1=0.005, critvalue2=0.05)
        (tn, fp, fn, tp) = confusion_matrix(list(map(int, new_df['lf{}'.format(lookforward)].values)), list(map(int, new_df['lb{}'.format(lookback)].values))).ravel()
        confusion = np.array([[tn, fp], [fn, tp]])
        print("For lookback {} and lookfoward {}".format(lookback, lookforward))
        print(confusion)
        recall = tp/(tp+fn)
        precision = tp / (tp + fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        print("Recall: {}".format(recall))
        print("Precision: {}".format(precision))
        print("Accuracy: {}".format(accuracy))
        print("F measure: {}".format(2*precision*recall/(precision+recall)))
        # plt.scatter(new_df['lb{}'.format(lookback)].values, new_df['lf{}'.format(lookforward)].values)
        # plt.xlabel("lookback")
        # plt.ylabel("lookforward")
        # plt.show()

    def convert_categories(self, df, critvalue1=0.005, critvalue2=0.05):
        list_of_columns = df.columns
        lookback_cols = list_of_columns[2:10]
        lookforward_cols = list_of_columns[10:]

        new_df = df.copy()
        bins1 = [0, critvalue1,  1]
        bins2 = [0, critvalue2, 1]
        labels = [1, 0]
        for i in lookback_cols:
            new_df[i] = pd.cut(new_df[i], bins=bins1, labels=labels)
        for i in lookforward_cols:
            new_df[i] = pd.cut(new_df[i], bins=bins2, labels=labels)
        return new_df



if __name__ == "__main__":
    x = DataManager()
    # # This code will just do it for one sector
    x.getOneSector(sector="Energy", fromDate="2011-01-01", toDate="2020-09-21")
    PC = PredictCoint(x.data)
    df = PC.combine_data()
    PC.confusion_analysis(df)
    #df = pd.read_csv("C:\\Users\\petem\\Trading\\Data\\cointegratedPair\\reorged\\2014-01-01.csv")
    # print("hi")
    # PC.saveCointegratedPairs(fromDate='2014-01-01', toDate='2017-01-01', lookback=True, cointOverArray=np.arange(10, 14, 2))
    # lookback_dict = PC.getDataGroup(lookback=True)
    # lookforward_dict = PC.getDataSingle(window=4, lookback=False)
    # print("hi")


