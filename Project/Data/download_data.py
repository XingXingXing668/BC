"""
Created on 12 Nov 2024

@author: Xingyi
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf
from pymongo import MongoClient

warnings.filterwarnings("ignore")

############################################
##          Mongo DB                    ##
############################################
def get_yfinance_data_Mongo(tickers, start=None, end=None):
    """
    Dynamically download historical stock data for a list of tickers, checking and auto-updating only new data in MongoDB.
    Parameters:
      tickers (list): Unique stock tickers in a list.
      start (str): Start date for historical data.
      end (str): End date for historical data.
    Returns:
      data (pd.DataFrame): Historical data downloaded from Yahoo Finance.
    """
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['financial_data']
        collection = db['historical_data']

        new_data = {}

        for ticker in tickers:
            # Check the latest date in MongoDB for the ticker
            latest_record = collection.find({"Ticker": ticker}).sort("Date", -1).limit(1)
            latest_date = None
            if latest_record.count() > 0:
                latest_date = latest_record[0]["Date"]
            start_date = pd.to_datetime(latest_date) + pd.Timedelta(days=1) if latest_date else start

            # Download new data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=end)
            if not data.empty:
                data['Ticker'] = ticker
                new_data[ticker] = data

        # Combine all ticker data into a single DataFrame
        if new_data:
            combined_data = pd.concat(new_data.values())
            combined_data.reset_index(inplace=True)
            combined_data.columns = [col if not isinstance(col, tuple) else col[1] for col in combined_data.columns]

            # Insert into MongoDB
            data_dict = combined_data.to_dict("records")
            collection.insert_many(data_dict)
            print(f"Data for {tickers} updated in MongoDB.")
        else:
            print("No new data to update.")
    except ValueError as ve:
        print('Download failed:', ve)
        data = None
    except Exception as e:
        print('An unexpected error occurred:', e)
        data = None

    return new_data


def get_yfinance_index_data_Mongo(tickers, start=None, end=None):
    """
    Download daily closing price data for indices and update MongoDB with the new data.
    Parameters:
      tickers (dict): mapping index names to Yahoo Finance tickers.
      start(str): optional start date for historical data. Defaults to first available date.
      end (str): optional end date for historical data. Defaults to today's date.
    Returns:
      combined_index_data (pd.DataFrame): combined daily closing prices for all indices.
    """
    end = end or datetime.today().strftime('%Y-%m-%d')
    client = MongoClient('mongodb://localhost:27017/')
    db = client['financial_data']
    collection = db['index_daily_data']
    df_list = []

    for name, ticker in tickers.items():
        # Check the latest date in MongoDB for this index
        latest_record = collection.find({"Index": name}).sort("Date", -1).limit(1)
        
        # Check if current date is a new start date for data download
        if latest_record.count() > 0:
            # If data exists, start from the day after the latest date
            latest_date = latest_record[0]["Date"]
            start_date = pd.to_datetime(latest_date) + pd.Timedelta(days=1)
        else:
            # If no data exists, use provided start or default to a broad range
            start_date = start or "2015-01-01"
        
        # Download new data
        index_data = yf.download(ticker, start=start_date, end=end, interval="1d")['Close']
        index_data.name = name

        if not index_data.empty:
            index_df = index_data.reset_index().rename(columns={'Close': 'Closing Price'})
            index_df['Index'] = name
            df_list.append(index_data)

            # Insert new index data into MongoDB
            collection.insert_many(index_df.to_dict("records"))
            print(f"Data for {name} updated in MongoDB.")
        else:
            print(f"No new data to update for {name}.")

    combined_index_data = pd.concat(df_list, axis=1) if df_list else pd.DataFrame()
    return combined_index_data



##############################################
##       Local file folder 	                ##
##############################################
def get_update_yf_pickle_local(tickers, start=None, end=None, filename="raw_daily_data.pkl"):
    """
    Update a pickle file offline by checking the last available date and downloading only new data.
    Ensures data is organized in the same vertical format as the original pickle file.
    Parameters:
      tickers (list): all unique tickers in a list.
      start(str): start date for historical data.
      end (str): end date for historical data. Defaults to today's date if None.
      filename (str): the name of the pickle file to update.
    Returns:
      updated_data (pd.DataFrame): updated data for all tickers.
    """
    try:
        end = end or datetime.today().strftime('%Y-%m-%d')
        existing_data = pd.read_pickle(filename) if os.path.exists(filename) else pd.DataFrame()
        new_data = {}

        for ticker in tickers:
            # Check start date for new data download
            if not existing_data.empty and ticker in existing_data['Ticker'].unique():
                latest_date = existing_data[existing_data['Ticker'] == ticker]['Date'].max()
                start_date = pd.to_datetime(latest_date) + pd.Timedelta(days=1)
            else:
                start_date = start or "2015-01-01"
            data = yf.download(ticker, start=start_date, end=end, group_by='ticker', progress= False)
            
            if not data.empty:
                data_vertical = data.stack(level=0).reset_index().rename(columns={'level_1': 'Ticker'})
                data_vertical.columns = [col if not isinstance(col, tuple) else col[1] for col in data_vertical.columns]
                data_vertical['Ticker'] = ticker  # Add Ticker column
                new_data[ticker] = data_vertical

        # Concatenate new data with existing data if any
        if new_data:
            new_data_df = pd.concat(new_data.values(), ignore_index=True)
            updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)
            updated_data.drop_duplicates(subset=['Date', 'Ticker'], keep='last', inplace=True)
        else:
            updated_data = existing_data  # No new data, use existing data

        updated_data.to_pickle(filename)
    except ValueError as ve:
        updated_data = None
    except Exception as e:
        updated_data = None

    return updated_data




def download_index_data_local(tickers=None, start="2015-01-01", end=None, filename="indices_daily_data.csv"):
    """
    Download daily closing price data for specified indices and save to a single CSV file.
    Parameters:
      tickers(dict): mapping of index names to Yahoo Finance tickers.
              Defaults to S&P 500, NASDAQ 100, and Russell 2000 indices.
      start (str): start date for historical data (default "2015-01-01").
      end   (str): end date for historical data. Defaults to today's date if None.
      filename (str): CSV file name to save the data (default "indices_daily_data.csv").
    Returns:
      combined_df (pd.DataFrame): combined daily closing prices for all indices.
    """
    
    # Set to default indices names if not provided
    if tickers is None:
        tickers = {
            "S&P 500": "^GSPC",
            "NASDAQ 100": "^NDX",
            "Russell 2000": "^RUT"
        }
    
    # Set end to today's date if not provided
    end = end or datetime.today().strftime('%Y-%m-%d')
    df_list = []
    
    for name, ticker in tickers.items():
        # Download data for the specified ticker
        index_data = yf.download(ticker, start=start, end=end, interval="1d")['Close']
        
        index_data.name = name
        df_list.append(index_data)

    # Combine all index Series into a single DataFrame with each index as a separate column
    combined_df = pd.concat(df_list, axis=1)
    combined_df.to_csv(filename)

    return combined_df





###################################################################
# Preprocessing data: 
# 1. Calculate log(price), daily return, etc
# 2. remove dup ticker (e.g., GOOG vs. GOOGL), in order to pair trade
####################################################################

def preprocess_ticker_stats(data):
    """
    Preprocess the ticker data for analysis by sorting, handling non-positive values,
    and calculating log prices and daily log returns.
    Parameters:
      data (pd.DataFrame): containing historical price data for multiple tickers with 'Ticker' and 'Date' columns.
    Returns:
      filtered_data (pd.DataFrame): with additional columns for log prices and daily log returns.
    """
    # Ensure the data is sorted by Date for proper log return calculation
    filtered_data = data.sort_values(by=['Ticker', 'Date'])
    
    # Calculate log prices and daily log returns
    filtered_data['Adj Close'] = np.where(filtered_data['Adj Close']<=0, filtered_data['Close'], filtered_data['Adj Close'])
    filtered_data['Log Adj Close'] = np.log(filtered_data['Adj Close'])
    filtered_data['Log Close'] = np.log(filtered_data['Close'])
    filtered_data['Daily Return'] = filtered_data.groupby('Ticker')['Adj Close'].pct_change()

    return filtered_data


def preprocess_ticker_replacement(data, ticker_mapping):
    """
    Preprocess the data by replacing duplicate tickers with the correct ticker.
    If both tickers appear on the same date, only keep the correct ticker.
    Parameters:
      data (pd.DataFrame): original stock data with columns ['Date', 'Ticker', ...].
      ticker_mapping (dict): mapping of original tickers to the correct ticker.
    Returns:
      data (pd.DataFrame): modified data with correct ticker names.
    """
    for correct_ticker, original_tickers in ticker_mapping.items():
        for original_ticker in original_tickers:
            if original_ticker != correct_ticker:
                # Find dates where both the correct and original tickers appear
                correct_dates = set(data[data['Ticker'] == correct_ticker]['Date'])
                original_dates = set(data[data['Ticker'] == original_ticker]['Date'])
                common_dates = correct_dates.intersection(original_dates)

                # Remove rows with the original ticker on common dates
                data = data[~((data['Ticker'] == original_ticker) & (data['Date'].isin(common_dates)))]
                data.loc[(data['Ticker'] == original_ticker) & ~(data['Date'].isin(correct_dates)), 'Ticker'] = correct_ticker
            
    return data



######################
##    Main          ##
######################
def main():
    # Step 1: load ticker files to get unique tickers
    source_file = 'SP500_tickers.csv'  # SP500
    sp500 = pd.read_csv(source_file)

    source_file = 'Russel2000_tickers.csv'  # Russel2000
    russel2000 = pd.read_csv(source_file)

    source_file = 'Ndq100_tickers.csv'  # Nasdaq100
    ndq100 = pd.read_csv(source_file)

    all_tickers = set(list(sp500['Ticker']) + list(russel2000['Ticker']) + list(ndq100['Ticker']))

    # Step 2: Update pickle/MongoDB with the tickers
    stock_data = get_update_yf_pickle_local(all_tickers, start="2024-01-01", end=None, filename="raw_daily_data.pkl")
    index_data = download_index_data_local(end=None, filename="indices_daily_data.csv")

    # Step 3: Preprocess stock data
    preprocessed_data = preprocess_ticker_stats(stock_data)

    ticker_mapping = {'GOOGL': ['GOOG', 'GOOGL'],
                      'FOXA':['FOXA', 'FOX'],
                      'BATRA':['BATRK','BATRA'],
                      'NWSA': ['NWS', 'NWSA'],
                      'CENTA': ['CENTA', 'CENT'],
                      'LILA': ['LILAK', 'LILA'],
                      'RUSHA': ['RUSHA', 'RUSHB'],
                     }
    preprocessed_data = preprocess_ticker_replacement(preprocessed_data, ticker_mapping).reset_index(drop=True)

    # Save the preprocessed data to a pickle file
    preprocessed_data.to_pickle("./preprocessed_daily_data.pkl")







