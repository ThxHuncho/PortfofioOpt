import numpy as np
import pandas as pd
import yfinance as yf

class DataTreatment:
    """
    Class for handling data: downloading, cleaning, and splitting.
    """

    @staticmethod
    def download_data(tickers, start, end):
        """
        Download historical price data using yfinance.

        :param tickers: list
            List of stock tickers.
        :param start: str
            Start date (YYYY-MM-DD).
        :param end: str
            End date (YYYY-MM-DD).
        :return: pd.DataFrame
            DataFrame of log returns with dates as the index.
        """
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        log_returns = np.log(data / data.shift(1)).dropna()
        return log_returns

    @staticmethod
    def split_data_by_percentage(data, train_size=0.8):
        """
        Split data into train and test sets based on a percentage.

        :param data: pd.DataFrame
            DataFrame of historical returns.
        :param train_size: float
            Proportion of data to use as training (default is 80%).
        :return: tuple
            Train and test DataFrames.
        """
        split_idx = int(len(data) * train_size)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        return train, test

    @staticmethod
    def split_data_by_date(data, split_date):
        """
        Split data into train and test sets based on a specific date.

        :param data: pd.DataFrame
            DataFrame of historical returns.
        :param split_date: str
            Date to split the data (format: YYYY-MM-DD).
        :return: tuple
            Train and test DataFrames.
        """
        train = data.loc[:split_date]
        test = data.loc[split_date:]
        return train, test

    @staticmethod
    def split_data_by_index(data, split_index):
        """
        Split data into train and test sets based on an index.

        :param data: pd.DataFrame
            DataFrame of historical returns.
        :param split_index: int
            Index at which to split the data.
        :return: tuple
            Train and test DataFrames.
        """
        train = data.iloc[:split_index]
        test = data.iloc[split_index:]
        return train, test
