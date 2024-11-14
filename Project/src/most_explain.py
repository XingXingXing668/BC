"""
Created on 12 Nov 2024

@author: Xingyi
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

from src.utils.data_mixin import DataMixin
from src.utils.plot_utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from turtledemo.penrose import start
from pykalman import KalmanFilter


warnings.filterwarnings("ignore")

class MostExplainStocks(DataMixin):
    def filter_data_by_chosen_index(self, daily_stock_return, daily_index_return, chosen_index, start_date=None, end_date=None):
        """
        Filters the stock and index data to include only stocks from a specified index  within a date range.
        Parameters:
            daily_stock_return (pd.DataFrame): DataFrame with stock daily return data where each column represents a stock ticker.
            daily_index_return (pd.DataFrame): DataFrame with index daily return data where each column represents an index ticker.
            chosen_index (str): The name of the index to filter stocks by (e.g., "SP500", "Russel2000", "Nasdaq100").
            start_date (str/None): The start date for the filter in 'YYYY-MM-DD' (optional).
            end_date (str/None): The end date for the filter in 'YYYY-MM-DD' (optional).
        Returns:
            filtered_stock_data (pd.DataFrames): Filtered stock
            filtered_index_data (pd.DataFrames): Filtered index
        """
        # Load tickers based on the chosen index
        if chosen_index == "SP500":
            tickers = self.get_sp500_tickers()
        elif chosen_index == "Russel2000":
            tickers = self.get_russel200_tickers()
        elif chosen_index == "Nasdaq100":
            tickers = self.get_ndq100_tickers()
        else:
            tickers = []
            print(f"Warning: Unrecognized index '{chosen_index}'. Returning empty ticker list.")

        available_tickers = [ticker for ticker in tickers if ticker in daily_stock_return.columns]
        filtered_stock_data = daily_stock_return[available_tickers].copy()

        # Map the chosen index to the column name in daily_index_return
        index_mapping = {
            "SP500": "SP500",
            "Russel2000": "Russel2000",
            "Nasdaq100": "Nasdaq100"
        }
        chosen_column = index_mapping.get(chosen_index, chosen_index)

        # Ensure the chosen index exists in daily_index_return
        if chosen_column not in daily_index_return.columns:
            raise ValueError(f"The chosen index '{chosen_column}' is not found in the daily index return data.")

        filtered_index_data = daily_index_return[[chosen_column]].copy()

        # Apply the date range filter to both stock and index data if dates are provided
        if start_date:
            filtered_stock_data = filtered_stock_data[filtered_stock_data.index >= start_date]
            filtered_index_data = filtered_index_data[filtered_index_data.index >= start_date]
        if end_date:
            filtered_stock_data = filtered_stock_data[filtered_stock_data.index <= end_date]
            filtered_index_data = filtered_index_data[filtered_index_data.index <= end_date]

        return filtered_stock_data, filtered_index_data

    def rolling_corr_index_stocks(self, daily_stock_return, daily_index_return, chosen_index, top_n=50, rolling_window=252):
        """
        Calculates the rolling correlation of each stock with the chosen index over a specified rolling window.
        Parameters:
            daily_stock_return (pd.DataFrame): DataFrame with daily returns of each stock.
            daily_index_return (pd.DataFrame): DataFrame with index daily returns.
            chosen_index (str): The column name of the chosen index in daily_index_return.
            top_n (int): Number of top stocks to select based on the highest average correlation.
            rolling_window (int): Rolling window size for correlation calculation.

        Returns:
            Y (pd.DataFrame): DataFrame containing the chosen index returns.
            X (pd.DataFrame): DataFrame containing the returns of the selected stocks.
        """
        # Calculate rolling mean for the chosen index
        rolling_index = daily_index_return[chosen_index].rolling(rolling_window, min_periods=int(0.8 * rolling_window)).mean()

        # Calculate rolling correlation of each stock with the rolling index
        rolling_correlations = pd.DataFrame(index=daily_stock_return.index, columns=daily_stock_return.columns)
        for stock in daily_stock_return.columns:
            rolling_correlations[stock] = daily_stock_return[stock].rolling(rolling_window, min_periods=int(0.8 * rolling_window)).corr(rolling_index)

        # Calculate rolling correlation
        mean_rolling_corr = rolling_correlations.mean(skipna=True)

        top_stocks = mean_rolling_corr.nlargest(top_n).index
        filtered_stock_returns = daily_stock_return[top_stocks]

        # Ensure Y and X are aligned after rolling correlations
        Y = daily_index_return[[chosen_index]].dropna()  # Drop NaNs in Y from rolling window
        X = filtered_stock_returns.reindex(Y.index)  # Align X to Y's index

        return Y, X


    def stepwise_top_stocks(self, X, Y, target_stocks=10):
        """
        Performs stepwise selection to identify top stocks that best explain the index returns.
        Parameters:
            X (pd.DataFrame): DataFrame containing the returns of the selected stocks (features).
            Y (pd.DataFrame): DataFrame containing the index returns (target).
            target_stocks (int): Number of top stocks to select based on highest R-squared improvement.
        Returns:
            selected_stocks (list): List of selected stock tickers that best explain the index returns.
            stock_betas (pd.Series): Beta values of the selected stocks in the final model.
        """
        # Forward fill, then backward fill any remaining NaNs in both X and Y
        X = X.ffill().bfill()
        Y = Y.ffill().bfill()

        
        selected_stocks = []
        remaining_stocks = list(X.columns)
        current_r2 = 0
        stock_betas = pd.Series(dtype='float64')

        # Stepwise selection
        while len(selected_stocks) < target_stocks and remaining_stocks:
            best_r2 = current_r2
            best_stock = None

            for stock in remaining_stocks:
                model = sm.OLS(Y, sm.add_constant(X[selected_stocks + [stock]])).fit()
                if model.rsquared > best_r2:
                    best_r2 = model.rsquared
                    best_stock = stock

            # Update selected and remaining stocks based on the best stock found
            if best_stock:
                selected_stocks.append(best_stock)
                remaining_stocks.remove(best_stock)
                current_r2 = best_r2
            else:
                break  # Exit if no improvement is made

        # Fit final model with selected stocks
        final_model = sm.OLS(Y, sm.add_constant(X[selected_stocks])).fit()
        stock_betas = final_model.params[1:]

        return selected_stocks, stock_betas


    def lasso_top_stocks(self, X, Y, target_stocks=10):
        """
        Uses Lasso regression with cross-validation to select the top stocks that best explain the index returns.
        Parameters:
            X (pd.DataFrame): DataFrame containing the returns of the selected stocks (features).
            Y (pd.DataFrame): DataFrame containing the index returns (target).
            target_stocks (int): Number of top stocks to select based on Lasso coefficients.
        Returns:
            selected_stocks (list): List of selected stock tickers that best explain the index returns.
            stock_betas (pd.Series): Non-zero beta values of the selected stocks.
        """
        # Forward fill, then backward fill any remaining NaNs in both X and Y to avoid excessive data loss
        X = X.ffill().bfill()
        Y = Y.ffill().bfill()

        # Ensure Y and X are array type for Lasso
        Y_vec = Y.squeeze().values
        X_vec = X.values

        # Use Lasso with cross-validation to find the optimal alpha
        lasso = LassoCV(cv=5).fit(X_vec, Y_vec)
        non_zero_coefs = lasso.coef_ != 0
        stock_betas = pd.Series(lasso.coef_[non_zero_coefs], index=X.columns[non_zero_coefs])

        # Limit to the top `target_stocks` based on absolute value of beta
        if len(stock_betas) > target_stocks:
            stock_betas = stock_betas.abs().nlargest(target_stocks)

        selected_stocks = list(stock_betas.index)

        return selected_stocks, stock_betas


    def kalman_filter_dynamic_betas(self, X, Y, target_stocks=10):
        """
        Applies Kalman filter to estimate dynamic betas (coefficients) for each stock in X with respect to Y.
        Parameters:
            X (pd.DataFrame): DataFrame with stock returns (50 columns, potentially with NaNs).
            Y (pd.Series): Series with index daily returns (target variable).
        Returns:
            dynamic_betas (pd.DataFrame): DataFrame of time-varying beta estimates for each stock.
        """

        n_stocks = X.shape[1]  # Number of stocks (features in X)
        n_timesteps = X.shape[0]  # Number of observations (dates)

        observation_covariance = 1.0
        transition_covariance = np.eye(n_stocks) * 0.01 # small value to allow for smooth beta changes over time
        transition_matrix = np.eye(n_stocks)
        initial_state_mean = np.zeros(n_stocks)
        initial_state_covariance = np.eye(n_stocks)* 10

        observation_matrices = []
        for i in range(n_timesteps):
            obs_matrix = X.iloc[i].fillna(0).values.reshape(1, -1)
            # Avoid dividing by zero by adding a small value (1e-10) to elements where needed
            obs_matrix = np.where(obs_matrix == 0, 1e-10, obs_matrix)
            observation_matrices.append(obs_matrix)

        # Initialize the Kalman filter
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrices,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance
        )


        state_means, _ = kf.filter(Y.values)
        dynamic_betas = pd.DataFrame(state_means, index=X.index, columns=X.columns)
        dynamic_betas_trimmed = dynamic_betas.iloc[130:]  # Adjust this number as needed
        avg_abs_betas = dynamic_betas_trimmed.replace(0, np.nan).abs().mean(skipna=True)
        selected_stocks = avg_abs_betas.nlargest(target_stocks).index

        return selected_stocks, dynamic_betas_trimmed[selected_stocks]



    def find_most_explain_stocks(self, daily_stock_return, daily_index_return, chosen_index=None,
                                 start_date=None, end_date=None, method="Kalman"):
        """
        Finds the top 10 stocks that most explain the chosen index using the specified method.
        Parameters:
            daily_stock_return (DataFrame): Daily returns of stocks.
            daily_index_return (DataFrame): Daily returns of indices.
            chosen_index (str): Index to analyze (e.g., "SP500", "Russel2000", "Nasdaq100").
            start_date (str): Start date for filtering data (optional).
            end_date (str): End date for filtering data (optional).
            method (str): Method to use for selection ("Stepwise", "Lasso", "PCA", or "Kalman").
        Returns:
            fig (go.Figure): figures with top 10 stocks and their corresponding beta.
        """
        # Filter stock and index data based on chosen index and date range
        filtered_stock_data, filtered_index_data = self.filter_data_by_chosen_index(daily_stock_return, daily_index_return,
                                                                               chosen_index, start_date=start_date, end_date=end_date)

        # Filter based on rolling correlation with the index to get the top 50 stocks
        Y, X = self.rolling_corr_index_stocks(filtered_stock_data, filtered_index_data, chosen_index, top_n=50, rolling_window=65)

        # Select top stocks based on the chosen method
        if method == "Stepwise":
            selected_stocks, stock_betas = self.stepwise_top_stocks(X, Y, target_stocks=10)
        elif method == "Lasso":
            selected_stocks, stock_betas = self.lasso_top_stocks(X, Y, target_stocks=10)
        elif method == "Kalman":
            selected_stocks, stock_betas = self.kalman_filter_dynamic_betas(X, Y, target_stocks=10)
        else:
            raise ValueError("Unsupported method specified.")
        fig = plot_betas(selected_stocks, stock_betas, method=method, index = chosen_index)
        
        return fig


