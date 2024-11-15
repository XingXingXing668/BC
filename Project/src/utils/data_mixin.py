import os
import pandas as pd
import yfinance as yf

class DataMixin:

    def load_data(self):
        """
        Load stock data to create a DataFrame.

        Returns:
        pd.DataFrame: Raw DataFrame for stocks.
        """
        current_dir = os.path.dirname(__file__)
        
        stock_data = pd.read_pickle('./Data/preprocessed_daily_data_small.pkl')
        ticker_sector = pd.read_csv("./Data/tickers_with_sector_industry.csv")
        
        # Merge the sector data with the main data
        stock_data = stock_data.merge(ticker_sector, on='Ticker', how='left')
        stock_data = stock_data.sort_values(by=['Date'])
        
        return stock_data
    
    def pivot_data(self, data, values='Close'):
        """
        Pivot stock data to create a DataFrame with dates as rows and tickers as columns, 
        allowing flexible selection of the values to pivot by.
        Parameters:
        data (pd.DataFrame): The input DataFrame containing stock data with columns: 
                            ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
           					'Log Adj Close', 'Log Close', 'Daily Return', 'Sector', 'Industry']
        values (str): The column name to use for the pivoted values. Defaults to 'Close'.
        Returns:
        pd.DataFrame: A pivoted DataFrame with dates as the index, tickers as columns, 
                      and the specified values as the data.
        """
        # Pivot the data with fixed 'Date' index and 'Ticker' columns, allowing variable 'values'
        pivoted_data = data.pivot(index='Date', columns='Ticker', values=values)
        
        pivoted_data.index = pd.to_datetime(pivoted_data.index)
        
        return pivoted_data

    def get_indices_daily_data(self):
        indices_daily_data = pd.read_csv('./Data/indices_daily_data.csv')
        indices_daily_data['Date'] = pd.to_datetime(indices_daily_data['Date'], utc=True)
        indices_daily_data.set_index('Date', inplace=True)
        return indices_daily_data

    def _merged_data(self, stock_data, indices_data):
        return stock_data.join(indices_data, how='outer')

    
    def _handle_missing_data(self, stock_data, indices_data):
        missing_data_summary = self._merged_data(stock_data, indices_data).isna().sum()
        print("Missing data summary:\n", pd.DataFrame(missing_data_summary, columns=['Count']).sort_values(by= ['Count'],ascending=False))

    def get_sp500_tickers(self):
        return pd.read_csv('./Data/SP500_tickers.csv')["Ticker"].tolist()

    def get_russel200_tickers(self):
        return pd.read_csv('./Data/Russel2000_tickers.csv')["Ticker"].tolist()

    def get_ndq100_tickers(self):
        return pd.read_csv('./Data/Ndq100_tickers.csv')["Ticker"].tolist()

    def load_raw_daily(self):
        raw_daily_data = pd.read_pickle('./Data/raw_daily_data_small.pkl')
        raw_daily_data['Daily Return'] = raw_daily_data.groupby('Ticker')['Adj Close'].pct_change()
        return raw_daily_data
