"""
Created on 12 Nov 2024

@author: Xingyi
"""

import threading
import warnings

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dash import dcc, html, ctx
# from dash_extensions import BackgroundJobManager, callback, Output
from dash.dependencies import Input, Output, State
from src.utils.data_mixin import DataMixin
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

warnings.filterwarnings("ignore")

class PairTradeAnalysis(DataMixin):
    
    #######################################################
    # Step 1: Calculate Correlation (within same sector)
    #######################################################
    def find_high_corr_pairs(self, data, correlation_threshold=0.8):
        """
        Function to find highly correlated pairs from daily returns DataFrame.
        Parameters:
            data (pd.DataFrame): Long table with columns: ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
               												'Log Adj Close', 'Log Close', 'Daily Return', 'Sector', 'Industry']
            correlation_threshold (float): Threshold for filtering high correlation.
        Returns:
            all_top_pairs (list): List of pairs with correlation above the threshold.
        """

        all_top_pairs = []
        # Calculate the correlation matrix for the sector
        for sector, sector_data in data.groupby('Sector'):
            
            pivot_data = sector_data.pivot(index='Date', columns='Ticker', values='Close')
            correlation_matrix = pivot_data.corr()

            # Extract pairs with correlation above the threshold (e.g., 0.8)
            top_pairs = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
            top_pairs = top_pairs[(top_pairs < 1.0) & (top_pairs > correlation_threshold)]

            top_pairs_list = [(pair[0], pair[1], sector) for pair in top_pairs.index]
            all_top_pairs.extend(top_pairs_list)

        return all_top_pairs


    #######################################################
    # Step 2: Cointegration Test
    #######################################################
    def check_cointegration_parallel(self, high_corr_pairs_list, pivot_prices, significance=0.1, n_jobs=-1):
        """
        Optimized function to check Engle-Granger cointegration test using parallel processing.
        Parameters:
            high_corr_pairs_list (list of tuples): List of (ticker1, ticker2, sector).
            pivot_prices (pd.DataFrame): DataFrame containing prices of all tickers.
            significance (float): Significance level for the p-value.
            n_jobs (int): Number of jobs for parallel processing (-1 to use all available cores).
        Returns:
            cointegrated_pairs (list of tuples): List of cointegrated pairs (ticker1, ticker2, sector).
        """
        def test_pair(ticker1, ticker2, sector):
            if ticker1 in pivot_prices.columns and ticker2 in pivot_prices.columns:
                combined_df = pivot_prices[[ticker1, ticker2]].dropna()
                if len(combined_df) > 0:
                    series1, series2 = combined_df[ticker1], combined_df[ticker2]
                    coint_test = coint(series1, series2)
                    p_value = coint_test[1]
                    if p_value < significance:
                        return (ticker1, ticker2, sector)
            return None

        # Run tests in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_pair)(ticker1, ticker2, sector) for ticker1, ticker2, sector in high_corr_pairs_list)

        # Filter out None results
        cointegrated_pairs = [pair for pair in results if pair is not None]

        return cointegrated_pairs


    #######################################################
    # Step 3: Identify Spread using OLS Regression
    #######################################################
    def calculate_ols_for_pair(self, ticker1, ticker2, sector, pivot_prices):
        """
        Calculate the OLS regression between two tickers to determine the hedge ratio and spread. Checks for mean-reverting characteristics in the spread using ADF and KPSS tests, and
        calculates mean reversion speed and half-life if the spread is stationary.
        Parameters:
            ticker1 (str): The name of ticker1 (dependent variable).
            ticker2 (str): The name of ticker2 (independent variable).
            sector (str): The sector to which the pair belongs.
            pivot_prices (pd.DataFrame): Pivoted DataFrame with prices, indexed by date, and tickers as columns.

        Returns:
            dict or None: Dictionary with regression results if the spread is mean-reverting, else None.
        """
        if ticker1 in pivot_prices.columns and ticker2 in pivot_prices.columns:
            combined_df = pivot_prices[[ticker1, ticker2]].dropna()
            if len(combined_df) > 0:
                series1, series2 = combined_df[ticker1], combined_df[ticker2]

                # OLS: series1 = alpha + beta * series2
                series2_with_const = sm.add_constant(series2)
                model = sm.OLS(series1, series2_with_const).fit()

                beta = model.params[ticker2]
                intercept = model.params['const']
                r_squared = model.rsquared

                # Only proceed if beta is non-zero
                if beta != 0:
                    # Calculate the spread
                    spread = series1 - (intercept + beta * combined_df[ticker2])

                    # Stationary Check,  Run ADF and KPSS tests
                    adf_test = adfuller(spread)
                    adf_p_value = adf_test[1]

                    # Run KPSS test with warning handling
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=InterpolationWarning)
                        kpss_test = kpss(spread, regression='c')
                        kpss_p_value = kpss_test[1]

                    # Check if the spread is strictly mean-reverting
                    if adf_p_value < 0.05 and kpss_p_value > 0.05:
                        # Estimate mean reversion speed using Ornstein-Uhlenbeck process
                        delta_spread = spread.diff().dropna()
                        spread_lagged = spread.shift(1).dropna()
                        delta_spread, spread_lagged = delta_spread.align(spread_lagged, join='inner')

                        if len(delta_spread) > 0 and len(spread_lagged) > 0:
                            spread_lagged_df = pd.DataFrame({'Lagged Spread': spread_lagged})
                            ou_model = sm.OLS(delta_spread, sm.add_constant(spread_lagged_df)).fit()
                            lambda_estimate = -ou_model.params['Lagged Spread']
                            half_life = np.log(2) / lambda_estimate if lambda_estimate > 0 else np.nan
                        else:
                            lambda_estimate = np.nan
                            half_life = np.nan

                        return {
                            'Ticker1(Y)': ticker1,
                            'Ticker2(X)': ticker2,
                            'Sector': sector,
                            'Spread_Vol': round(spread.std(), 3),
                            'Is_Spread_Stationary': 1,
                            'OLS_hedge_ratio(per ticker1)': round(beta, 3),
                            'OLS_R2': round(r_squared, 3),
                            'Mean_Reversion_Speed(lambda)': round(lambda_estimate, 3),
                            'half_life(days)': round(half_life, 3),
                        }
        return None


    def calculate_ols_spread_for_pairs(self, cointegrated_pairs, pivot_prices, n_jobs=-1):
        """
        Calculate OLS spreads for cointegrated pairs using parallel processing.
        Parameters:
            cointegrated_pairs (list of tuples): List of tuples containing (ticker1, ticker2, sector) for each pair.
            pivot_prices (pd.DataFrame): Pivoted DataFrame with prices, indexed by date, and tickers as columns.
            n_jobs (int): Number of jobs for parallel processing. Defaults to -1 (uses all processors).
        Returns:
            results_df (pd.DataFrame): DataFrame with regression and mean reversion results for all pairs, sorted by Spread_Vol.
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.calculate_ols_for_pair)(ticker1, ticker2, sector, pivot_prices)
            for ticker1, ticker2, sector in cointegrated_pairs
        )

        # Filter out None results and convert to DataFrame
        results = [res for res in results if res is not None]
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Spread_Vol'], ascending=False)

        return results_df


    #######################################################
    # Step 4: Identify Spread Using a Kalman Filter
    #######################################################

    def estimate_kalman_for_pair(self, ticker1, ticker2, sector, pivot_prices):
        """
        Estimate the hedge ratio and spread for a pair of tickers using a Kalman filter. Conducts stationarity tests on the spread and calculates mean reversion speed
        and half-life if the spread is stationary.

        Parameters:
            ticker1 (str): The name of ticker1 (dependent variable).
            ticker2 (str): The name of ticker2 (independent variable).
            sector (str): The sector to which the pair belongs.
            pivot_prices (pd.DataFrame): Pivoted DataFrame with prices, indexed by date, and tickers as columns.

        Returns:
            dict or None: Dictionary with Kalman filter results if the spread is mean-reverting, else None.
        """
        # Ensure both tickers exist in DataFrame
        if ticker1 in pivot_prices.columns and ticker2 in pivot_prices.columns:
            combined_df = pivot_prices[[ticker1, ticker2]].dropna()
            if len(combined_df) > 0:
                y, x = combined_df[ticker1].values, combined_df[ticker2].values

                obs_mat = sm.add_constant(x, prepend=False)[:, np.newaxis]

                # Set up the Kalman filter
                kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                                  initial_state_mean=np.ones(2),
                                  initial_state_covariance=np.ones((2, 2)),
                                  transition_matrices=np.eye(2),
                                  observation_matrices=obs_mat,
                                  observation_covariance=10 ** 2,
                                  transition_covariance=0.01 ** 2 * np.eye(2))

                
                state_means, _ = kf.filter(y)

                # Apply burn-in period (3 months)
                burn_in = 65
                state_means = state_means[burn_in:]
                y = y[burn_in:]
                x = x[burn_in:]

                # Extract alpha and beta from the state means
                beta_series = pd.Series(state_means[:, 1], index=combined_df.index[burn_in:])
                alpha_series = pd.Series(state_means[:, 0], index=combined_df.index[burn_in:])
                spread = pd.Series(y - (beta_series * x + alpha_series), index=combined_df.index[burn_in:])

                # Stationary Check,  Run ADF and KPSS tests
                adf_test = adfuller(spread)
                adf_p_value = adf_test[1]

                # Run KPSS test with warning handling
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=InterpolationWarning)
                    kpss_test = kpss(spread, regression='c')
                    kpss_p_value = kpss_test[1]

                # Check if the spread is strictly mean-reverting
                if adf_p_value < 0.05 and kpss_p_value > 0.05:
                    # Mean reversion speed and half-life calculation
                    delta_spread = spread.diff().dropna()
                    spread_lagged = spread.shift(1).dropna()
                    delta_spread, spread_lagged = delta_spread.align(spread_lagged, join='inner')

                    if len(delta_spread) > 0 and len(spread_lagged) > 0:
                        spread_lagged_df = pd.DataFrame({'Lagged Spread': spread_lagged})
                        ou_model = sm.OLS(delta_spread, sm.add_constant(spread_lagged_df)).fit()
                        lambda_estimate = -ou_model.params['Lagged Spread']
                        half_life = np.log(2) / lambda_estimate if lambda_estimate > 0 else np.nan
                    else:
                        lambda_estimate = np.nan
                        half_life = np.nan

                    return {
                        'Ticker1(Y)': ticker1,
                        'Ticker2(X)': ticker2,
                        'Sector': sector,
                        'Spread_Vol': round(spread.std(), 3),
                        'Is_Spread_Stationary': 1,
                        'Kalman_avg_hedge_ratio(per ticker1)': round(beta_series.mean(), 3),
                        'Kalman_hedge_ratio_range': f"{round(beta_series.min(), 3)} - {round(beta_series.max(), 3)}",
                        'Mean_Reversion_Speed(lambda)': round(lambda_estimate, 3),
                        'half_life(days)': round(half_life, 3),
                    }
        return None


    def estimate_kalman_params_for_pairs(self, cointegrated_pairs, pivot_prices, n_jobs=-1):
        """
        Estimate Kalman filter parameters for multiple cointegrated pairs in parallel.

        Parameters:
            cointegrated_pairs (list of tuples): List of tuples containing (ticker1, ticker2, sector) for each pair.
            pivot_prices (pd.DataFrame): Pivoted DataFrame with prices, indexed by date, and tickers as columns.
            n_jobs (int): Number of jobs for parallel processing. Defaults to -1 (uses all processors).

        Returns:
            results_df (pd.DataFrame): DataFrame with Kalman filter and mean reversion results for all pairs, sorted by Spread_Vol.
        """
        # Run the Kalman filter estimation in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.estimate_kalman_for_pair)(ticker1, ticker2, sector, pivot_prices)
            for ticker1, ticker2, sector in cointegrated_pairs
        )

        # Filter out None results and create a DataFrame
        results = [res for res in results if res is not None]
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by=['Spread_Vol'], ascending=False)
        
        return results_df


    #######################################################
    # Wrap them up
    #######################################################
    def run_pairs_trading_workflow(self, data, start_date, end_date, pivot_column='Log Close'):
        """
        Runs the entire pairs trading workflow from filtering data to calculating OLS or Kalman filter spreads.
        Allows specification of the column to pivot on (e.g., 'Log Close' or 'Close').

        Parameters:
            data (pd.DataFrame): The input DataFrame with stock data containing 'Date', 'Ticker', and other value columns.
            start_date (str or pd.Timestamp): The start date for filtering the data.
            end_date (str or pd.Timestamp): The end date for filtering the data.
            pivot_column (str): The column name to use for pivoting. Defaults to 'Log Close'.

        Returns:
            dict: A dictionary containing DataFrames of OLS and Kalman filter results.
        """
        # Ensure 'Date' column is in datetime format and naive (no timezone)
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        start_date = pd.to_datetime(start_date).tz_localize(None)
        end_date = pd.to_datetime(end_date).tz_localize(None)

        # Initialize progress bar
        with tqdm(total=100, desc="Pairs Trading Workflow") as pbar:
            # Filter data within time range
            data = data.sort_values(by=['Date'])
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            pbar.update(10)  # 10% done after data filtering

            # Find highly correlated pairs
            high_corr_pairs = self.find_high_corr_pairs(data, correlation_threshold=0.9)
            pbar.update(10)  # 20% done after finding correlated pairs

            # Check cointegration of the pairs
            pivot_data = data.pivot(index='Date', columns='Ticker', values=pivot_column)
            pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)
            cointegrated_pairs = self.check_cointegration_parallel(high_corr_pairs, pivot_data, significance=0.05, n_jobs=-1)
            pbar.update(30)  # 50% done after cointegration check

            if not cointegrated_pairs:
                pbar.update(50)  # Complete the progress bar
                print(
                    "No valid cointegrated pairs found for the selected date range and criteria. Analysis cannot be performed.")
                return "No valid cointegrated pairs found for the selected date range and criteria. Analysis cannot be performed."

            # Calculate spreads for OLS and Kalman
            ols_results_df = self.calculate_ols_spread_for_pairs(cointegrated_pairs, pivot_data, n_jobs=-1)
            pbar.update(20)  # 70% done after OLS calculations

            cointegrated_pairs = list(
                ols_results_df[['Ticker1(Y)', 'Ticker2(X)', 'Sector']].itertuples(index=False, name=None))
            kalman_results_df = self.estimate_kalman_params_for_pairs(cointegrated_pairs, pivot_data, n_jobs=-1)
            pbar.update(30)  # 100% done after Kalman filter calculations

        return {
            "OLS": ols_results_df,
            "Kalman": kalman_results_df
        }

