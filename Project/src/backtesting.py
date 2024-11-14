"""
Created on 12 Nov 2024

@author: Xingyi
"""


import threading
import webbrowser

import dash
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from dash import dcc, html

from src.utils.data_mixin import DataMixin
from src.utils.plot_utils import *


class BackTesting(DataMixin):
    def __init__(self):
        super().__init__()

    def _calculate_spread(self, ticker1, ticker2, pivot_prices, method="OLS"):
        """
        Calculate the spread between two tickers using the specified method.
        Parameters:
            ticker1 (str): The name of the first ticker.
            ticker2 (str): The name of the second ticker.
            pivot_prices (pd.DataFrame): DataFrame with prices for each ticker as columns.
            method (str): Method to calculate spread, "OLS" or "Kalman".

        Returns:
            spread (pd.Series): The calculated spread series.
            beta_series (pd.Series):  time-varying beta estimates
        """
        if ticker1 not in pivot_prices.columns or ticker2 not in pivot_prices.columns:
            raise ValueError("One or both tickers not found in pivot prices DataFrame")

        # Get the tickers' price series
        ticker1_prices = pivot_prices[ticker1].dropna()
        ticker2_prices = pivot_prices[ticker2].dropna()
        ticker1_prices, ticker2_prices = ticker1_prices.align(ticker2_prices, join='inner')

        # OLS: ticker1 = alpha + beta * ticker2
        if method == "OLS":
            ticker2_with_const = sm.add_constant(ticker2_prices)
            model = sm.OLS(ticker1_prices, ticker2_with_const).fit()

            # hedge ratio (beta) and intercept (alpha)
            beta = model.params[ticker2]
            alpha = model.params['const']

            # Spread = ticker1 - (alpha + beta * ticker2)
            spread = ticker1_prices - (alpha + beta * ticker2_prices)
            return spread, beta

        elif method == "Kalman":
            y = ticker1_prices.values
            x = ticker2_prices.values

            obs_mat = sm.add_constant(x, prepend=False)[:, np.newaxis]
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                              initial_state_mean=np.zeros(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=10 ** 2,
                              transition_covariance=0.01 ** 2 * np.eye(2))

            state_means, _ = kf.filter(y)

            # Only keep stable betas after burn-in
            burn_in = 65
            state_means = state_means[burn_in:]
            y = y[burn_in:]
            x = x[burn_in:]

            # Extract dynamic alpha and dynamic beta
            beta_series = pd.Series(state_means[:, 1], index=ticker1_prices.index[burn_in:])
            alpha_series = pd.Series(state_means[:, 0], index=ticker1_prices.index[burn_in:])

            # Calculate the spread
            spread = pd.Series(y - (beta_series * x + alpha_series), index=ticker1_prices.index[burn_in:])
            return spread, beta_series

        else:
            raise ValueError("Invalid method. Choose 'OLS' or 'Kalman'.")

    #######################################################
    #Step 2: Generate trading signals/positions:
    #######################################################
    def generate_signals_rolling(self, spread, entry_thres, exit_thres, window=130, duration_cap=30, min_hold_days=3):
        """
        Generates trading signals based on rolling entry and exit thresholds, with a maximum duration cap, and minimum holding days.
        Parameters:
            spread (pd.Series): the spread between the two tickers in the pair.
            entry_thres (float): the entry threshold
            exit_thres (float): the exit threshold
            window (int): the rolling window size for calculating the mean and standard deviation, default is 130 (typically half trading year).
            duration_cap (int): the maximum number of days to hold a position before it is automatically exited.
            min_hold_days (int): the minimum number of days to hold a position before it is automatically exited.

        Returns:
            signals (pd.DataFrame): a DataFrame with columns:
                'position': int, indicating the trading position for each day:
                    position = 1: Enter a long position on the spread (buy ticker1, short ticker2).
                    position = -1: Enter a short position on the spread (short ticker1, buy ticker2).
                    position = 0: No position (hold cash or exit previous position).

                'trade': int, indicating the changes in the position on each day:
                    trade = 1: Entering a long position.
                    trade = 2: Exiting a long position.
                    trade = -1: Entering a short position.
                    trade = -2: Exiting a short position.
                    trade = 0: No new entry or exit signal on that day.
        """
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['position'] = 0

        # Calculate rolling thresholds
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        # Entry/Exit thresholds
        upper_entry = rolling_mean + entry_thres * rolling_std
        lower_entry = rolling_mean - entry_thres * rolling_std
        upper_exit = rolling_mean + exit_thres * rolling_std
        lower_exit = rolling_mean - exit_thres * rolling_std

        # Entry and exit signals
        entry_long = (spread < lower_entry)
        entry_short = (spread > upper_entry)
        exit_long = (spread > lower_exit)
        exit_short = (spread < upper_exit)

        # Calculate position
        signals['position'] = np.where(entry_long, 1, np.where(entry_short, -1, 0))
        signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)

        # Logic with minimum hold duration constraint
        hold_counter = np.zeros(len(signals))
        for i in range(1, len(signals)):
            if signals['position'].iloc[i] != 0:
                hold_counter[i] = hold_counter[i-1] + 1
            else:
                hold_counter[i] = 0

            if signals['position'].iloc[i] == 1 and exit_long.iloc[i]:
                if hold_counter[i] >= min_hold_days:
                    signals['position'].iloc[i] = 0
            if signals['position'].iloc[i] == -1 and exit_short.iloc[i]:
                if hold_counter[i] >= min_hold_days:
                    signals['position'].iloc[i] = 0

        # Calculate the position changes on each day
        signals['trade'] = 0
        signals.loc[(signals['position'].shift(1) == 0) & (signals['position'] == 1), 'trade'] = 1  # Enter long
        signals.loc[(signals['position'].shift(1) == 1) & (signals['position'] == 0), 'trade'] = 2  # Exit long
        signals.loc[(signals['position'].shift(1) == 0) & (signals['position'] == -1), 'trade'] = -1  # Enter short
        signals.loc[(signals['position'].shift(1) == -1) & (signals['position'] == 0), 'trade'] = -2  # Exit short

        # Enforce duration cap
        trade_durations = signals['position'].groupby((signals['position'] != signals['position'].shift()).cumsum()).cumcount()
        signals['position'] = np.where(trade_durations >= duration_cap, 0, signals['position'])

        return signals



    #######################################################
    #Step 3: Backtesting returns using the signals
    #######################################################

    def backtest_performance_with_signals(self, price_pivot_data, ticker1, ticker2, signals, transaction_cost=0.0001):
        """
        Backtest signals DataFrame to get strategy returns. Consider the transaction cost.
        Parameters:
            price_pivot_data (pd.DataFrame): with daily prices for all tickers, each column representing a ticker.
            ticker1 (str): the name of ticker1
            ticker2 (str): the name of ticker2
            signals (pd.DataFrame): containing 'position' and 'trade' columns for the strategy.
            transaction_cost (float): cost per trade as a proportion of the position size (default: 0.0001 or 0.01%).

        Returns:
            signals (pd.DataFrame): with added columns for strategy returns, cumulative return, and annualized return.
        """
        # Calculate daily returns for each ticker (percentage change)
        ticker1_returns = price_pivot_data[ticker1].pct_change().fillna(0)
        ticker2_returns = price_pivot_data[ticker2].pct_change().fillna(0)

        # Calculate daily strategy returns
        signals['strategy_return'] = signals['position'].shift(1) * (ticker1_returns - ticker2_returns)
        signals['strategy_return'] -= transaction_cost * signals['trade'].abs()
        signals = signals[~signals['strategy_return'].isin([float('inf'), -float('inf')])]

        # Calculate cumulative return
        signals['cumulative_return'] = (1 + signals['strategy_return']).cumprod() - 1  # Cumulative return as a growth factor

        return signals



    ##############################################################################################################
    ## Step 4: Calculate Comprehensive Performance Metrics (Sharpe, Sortino, Drawdown, etc)
    ##############################################################################################################
    def evaluate_performance(self, signals_return):
        """
        Evaluates the performance of the pairs trading strategy. E.g., Sharpe Ratio, Sortino, etc
        Parameters:
            signals_return (pd.DataFrame): Daily returns of the pairs trading strategy.
        Returns:
            metrics (dict): A dictionary containing eight performance metrics.
        """

        strategy_returns = signals_return['strategy_return']

        # Metric 1: Annual Return (assuming 252 trading days)
        ann_return = strategy_returns.mean() * 252

        # Metric 2: Annual Vol (assuming 252 trading days)
        ann_vol = strategy_returns.std() * np.sqrt(252)

        # Metric 3: Annual Sharpe Ratio (assuming 252 trading days)
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()

        # Metric 4: Sortino Ratio (assuming 252 trading days)
        downside_returns = strategy_returns[strategy_returns < 0]  # Only negative returns
        downside_std = downside_returns.std()
        sortino_ratio = np.sqrt(252) * strategy_returns.mean() / downside_std if downside_std != 0 else np.nan


        # Metric 5: Gain-to-Pain Ratio
        total_profit = strategy_returns.sum()
        total_loss = abs(strategy_returns[strategy_returns < 0]).sum()
        gain_to_pain_ratio = total_profit / total_loss if total_loss != 0 else np.nan


        # Metric 6: Maximum Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdown.max()
        max_dd_index = drawdown.idxmax()

        # Metric 7: Drawdown-to-Volatility Ratio using rolling volatility at max drawdown date
        rolling_volatility = strategy_returns.rolling(window=130).std()  # Rolling volatility over the last 130 days
        volatility_at_max_dd = rolling_volatility.loc[max_dd_index]  # Volatility at the max drawdown index
        if volatility_at_max_dd in (np.nan, 0, 0.0, None) or isinstance(volatility_at_max_dd, pd._libs.missing.NAType):
            volatility_at_max_dd = ann_vol
        drawdown_to_vol_ratio = max_drawdown / volatility_at_max_dd if volatility_at_max_dd != 0 else np.nan

        # Metric 8: Calmar Ratio (Return-to-DD Ratio)
        calmar_ratio = ann_return / max_drawdown if max_drawdown != 0 else np.nan


        metrics = {
            'Ann. Return': round(ann_return,3),
            'Ann. Vol': round(ann_vol, 3),
            'Ann. Sharpe Ratio': round(sharpe_ratio,3),
            'Ann. Sortino Ratio': round(sortino_ratio,3),
            'Gain-to-Pain Ratio': round(gain_to_pain_ratio,3),
            'Max Drawdown': round(max_drawdown,3),
            'Drawdown-to-Volatility Ratio': round(drawdown_to_vol_ratio,3),
            'Calmar Ratio (Return-to-DD)': round(calmar_ratio,3)
        }
        return metrics


    ######################################################################
    ## Step 5: Bayesian Optimization to find optimized entry_thres and exit_thres
    ######################################################################
    def pair_backtest_performance(self, spread, entry_thres, exit_thres, price_pivot_data, ticker1, ticker2, transaction_cost=0.0001):
        """
        Run the pairs trading strategy for given entry_thres and exit_thres.
        Parameters:
            spread (pd.Series): the spread data based on the Kalman filter or OLS.
            entry_thres (float): the entry threshold.
            exit_thres (float): the exit threshold.
            price_pivot_data (pd.DataFrame): prices for the tickers, each column is a ticker
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            transaction_cost (float): transaction cost for trading (default is 0.0001).

        Returns:
            performance_metrics (dict): performance metrics for the strategy.
        """
        signals = self.generate_signals_rolling(spread, entry_thres, exit_thres, window=252, duration_cap=30, min_hold_days=5)
        signals_return = self.backtest_performance_with_signals(price_pivot_data, ticker1, ticker2, signals, transaction_cost=0.0001)
        performance_metrics = self.evaluate_performance(signals_return)

        return performance_metrics


    def objective(self, params, spread, price_pivot_data, ticker1, ticker2, criterion='sharpe_ratio'):
        """
        Objective function for optimizing entry_thres and exit_thres based on different performance criteria.
        Parameters:
            params (list): entry_thres and exit_thres values.
            spread (pd.Series): the spread data based on the Kalman filter or OLS.
            price_pivot_data (pd.DataFrame): prices for the tickers, each column is a ticker
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            criterion (str): The performance metric to optimize (e.g., sharpe ratio, max_drawdown, etc)

        Returns:
            score (float): the score based on the chosen criterion (negative for minimization).
        """
        entry_thres, exit_thres = params
        performance_metrics = self.pair_backtest_performance(spread, entry_thres, exit_thres, price_pivot_data, ticker1, ticker2)

        # Extract performance metrics
        ann_return = performance_metrics['Ann. Return']
        ann_vol = performance_metrics['Ann. Vol']
        sharpe_ratio = performance_metrics['Ann. Sharpe Ratio']
        sortino_ratio = performance_metrics['Ann. Sortino Ratio']
        gain_to_pain_ratio = performance_metrics['Gain-to-Pain Ratio']
        max_drawdown = performance_metrics['Max Drawdown']
        drawdown_to_vol_ratio = performance_metrics['Drawdown-to-Volatility Ratio']
        calmar_ratio = performance_metrics['Calmar Ratio (Return-to-DD)']

        if criterion == 'sharpe_ratio':
            score = sharpe_ratio
        elif criterion == 'ann_return':
            score = ann_return
        elif criterion == 'sortino_ratio':
            score = sortino_ratio
        elif criterion == 'gain_to_pain_ratio':
            score = gain_to_pain_ratio
        elif criterion == 'calmar_ratio':
            score = calmar_ratio
        # Minimize max drawdown (use negative)
        elif criterion == 'max_drawdown':
            score = -max_drawdown
        # Minimize Drawdown-to-Volatility ratio
        elif criterion == 'drawdown_to_vol_ratio':
            score = -drawdown_to_vol_ratio
        else:
            raise ValueError("Unsupported criterion specified.")

        return score


    def calculate_k_bounds(self, spread):
        """
        Calculate entry_thres and exit_thres bounds for a given spread by normalizing.
        Parameters:
            spread (pd.Series): the spread data based on the Kalman filter or OLS.

        Returns:
            entry_thres_bounds: tuple, bounds for entry_thres
            exit_thres_bounds: tuple, bounds for exit_thres
        """

        mean_spread = spread.mean()
        std_spread = spread.std()
        normalized_spread = (spread - mean_spread) / std_spread

        # Define fixed bounds on the normalized spread
        entry_thres_bounds = (0.5, 4.0)  # 0.5 to 4 standard deviations
        exit_thres_bounds = (0.1, 2.0)   # 0.1 to 2 standard deviations
        real_entry_thres = (mean_spread + 0.5 * std_spread, mean_spread + 4 * std_spread)
        real_exit_thres = (mean_spread + 0.1 * std_spread, mean_spread +  2 * std_spread)

        return real_entry_thres, real_exit_thres


    def calculate_dynamic_bounds(self,spread):
        """
        Calculate dynamic bounds for entry_thres and exit_thres based on spread statistics.
        Parameters:
            spread (pd.Series): the spread data based on the Kalman filter or OLS.

        Returns:
            entry_thres_bounds: tuple, bounds for entry_thres
            exit_thres_bounds: tuple, bounds for exit_thres
        """
        mean_spread = spread.mean()
        std_spread = spread.std()

        # Entry bounds around 2-5 standard deviations
        entry_thres_bounds = (mean_spread + 2 * std_spread, mean_spread + 5 * std_spread)
        # Exit bounds around 0.5-2 standard deviations
        exit_thres_bounds = (max(0, mean_spread + 0.5 * std_spread), max(0, mean_spread + 2 * std_spread))

        return entry_thres_bounds, exit_thres_bounds


    def optimize_k(self, spread, price_pivot_data, ticker1, ticker2, criterion='sharpe_ratio'):
        """
        Optimize entry_thres and exit_thres using Bayesian optimization.
        Parameters:
            spread (pd.Series): the spread data based on the Kalman filter or OLS.
            price_pivot_data (pd.DataFrame): prices for the tickers, each column is a ticker
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            criterion (str): The performance metric to optimize (e.g., sharpe ratio, max_drawdown, etc)

        Returns:
            entry_thres (float): best entry_thres.
            exit_thres (float): best exit_thres.
        """

        entry_thres_bounds, exit_thres_bounds = self.calculate_dynamic_bounds(spread)
        pbounds = {
            'entry_thres': entry_thres_bounds,
            'exit_thres': exit_thres_bounds
        }

        # Bayesian optimizer
        optimizer = BayesianOptimization(
            f=lambda entry_thres, exit_thres: self.objective([entry_thres, exit_thres], spread, price_pivot_data, ticker1, ticker2, criterion),
            pbounds=pbounds,
            random_state=1,
            verbose=0,
        )
        optimizer.maximize(
            init_points=10,
            n_iter=30 
        )

        # Get the best parameters found
        best_params = optimizer.max
        entry_thres = best_params['params']['entry_thres']
        exit_thres = best_params['params']['exit_thres']
        metric_value = best_params['target']

        return round(entry_thres,3), round(exit_thres,3)


    def optimize_and_store_results(self, spread, price_pivot_data, ticker1, ticker2, performance_metrics_list):
        """
        Optimizes entry and exit thresholds for each performance metric, stores results in a DataFrame.
        Parameters:
            spread (pd.Series): the spread data based on the Kalman filter or OLS.
            price_pivot_data (pd.DataFrame): prices for the tickers, each column is a ticker
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            performance_metrics_list (list): list of performance metrics to optimize.

        Returns:
            results_df (pd.DataFrame): results with each metric as the objective criterion.
        """
        results = {}
        for criterion in performance_metrics_list:
            print(f"Optimizing for {criterion}...")
            best_entry_thres, best_exit_thres = self.optimize_k(spread, price_pivot_data, ticker1, ticker2, criterion=criterion)

            # Calculate performance metrics
            performance_metrics = self.pair_backtest_performance(spread, best_entry_thres, best_exit_thres, price_pivot_data, ticker1, ticker2)
            results[criterion] = {
                'entry_thres': best_entry_thres,
                'exit_thres': best_exit_thres,
                **performance_metrics
            }

        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.index.name = "Objective Criterion"
        return results_df


    #######################################################
    # Wrap them up
    #######################################################
    def backtest_pair_trading_pipeline(self, ticker1, ticker2, price_pivot_data, spread_type='Kalman',
                                       entry_thres=1.5, exit_thres=0.5,
                                       transaction_cost=0.0001, window=252, duration_cap=30, min_hold_days=3,
                                       criterion='sharpe_ratio', optimize=False, performance_metrics_list=None):
        """
        Pipeline for pair trading backtest, with optional optimization.

        Parameters:
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            price_pivot_data (pd.DataFrame): prices for the tickers, each column is a ticker
            spread_type (str): Choice of 'OLS' or 'Kalman' spread calculation.
            entry_thres (float): the entry threshold.
            exit_thres (float): the exit threshold.
            transaction_cost (float): transaction cost for trading (default is 0.0001).
            window (int): the rolling window size for calculating the mean and standard deviation, default is 130 (typically half trading year).
            duration_cap (int): the maximum number of days to hold a position before it is automatically exited.
            min_hold_days (int): the minimum number of days to hold a position before it is automatically exited.
            criterion (str): The performance metric to optimize (e.g., sharpe ratio, max_drawdown, etc) if optimize is True.
            optimize (Boolean): determine if Bayesian optimization should be used.
            performance_metrics_list (list): List of metrics to optimize if optimize is True.

        Returns:
            results_df (pd.DataFrame): results with performance metrics for the optimized thresholds.
        """
        
        if spread_type == 'OLS':
            spread, beta = self._calculate_spread(ticker1,ticker2, price_pivot_data, method="OLS")
        elif spread_type == 'Kalman':
            spread, beta = self._calculate_spread(ticker1,ticker2, price_pivot_data, method="Kalman")
        else:
            raise ValueError("Invalid spread type. Choose 'OLS' or 'Kalman'.")

        # Generate trading signals, calculate signal return, evaluate initial performance metric
        signals = self.generate_signals_rolling(spread, entry_thres, exit_thres, window=window, duration_cap=duration_cap, min_hold_days=min_hold_days)
        signals_return = self.backtest_performance_with_signals(price_pivot_data, ticker1, ticker2, signals, transaction_cost)
        initial_metrics = self.evaluate_performance(signals_return)

        # If optimize is True, optimize the thresholds for each performance metric
        if optimize and performance_metrics_list:
            results_df = self.optimize_and_store_results(spread, price_pivot_data, ticker1, ticker2, performance_metrics_list)
        else:
            results_df = pd.DataFrame([initial_metrics], index=['Initial Performance'])

        return results_df



    #######################################################
    # Sensitivity check: varying entry/exit threshold
    #######################################################

    # Generate Sharpe ratio data function
    def calculate_sharpe_ratio(self, spread, price_pivot_data, ticker1, ticker2, varying_values, fixed_value,
                               vary_parameter='entry_thres', criterion='sharpe_ratio'):
        """
        Calculates Sharpe ratios for a range of either entry_thres or exit_thres values, with the other parameter fixed.
        Parameters:
            spread (pd.Series): the spread data based on the Kalman filter or OLS.
            price_pivot_data (pd.DataFrame): prices for the tickers, each column is a ticker
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            varying_values (list or np.array): Range of values for the parameter being varied.
            fixed_value (float): Fixed value of the other parameter.
            vary_parameter (str): Specifies which parameter to vary ('entry_thres' or 'exit_thres').
            criterion (str): The performance metric to optimize (default is 'sharpe_ratio').

        Returns:
            pd.DataFrame: DataFrame with the varying parameter and corresponding Sharpe ratios.
        """
        sharpe_ratios = []
        for value in varying_values:
            if vary_parameter == 'entry_thres':
                entry_thres, exit_thres = value, fixed_value
            elif vary_parameter == 'exit_thres':
                entry_thres, exit_thres = fixed_value, value
            else:
                raise ValueError("vary_parameter must be 'entry_thres' or 'exit_thres'")

            performance_metrics = self.pair_backtest_performance(spread, entry_thres, exit_thres, price_pivot_data, ticker1, ticker2)
            sharpe_ratios.append(performance_metrics['Ann. Sharpe Ratio'])

        #  Output dataframe
        return pd.DataFrame({vary_parameter: varying_values, 'Sharpe Ratio': sharpe_ratios})


    def show_board(self, ticker1, ticker2, initial_entry_thres, initial_exit_thres, fixed_exit_val=1.0, fixed_entry_val=2.0):
        """
        Show the dashboard
        Parameters:
            ticker1 (str): name of ticker1
            ticker2 (str): name of ticker2
            initial_entry_thres (float): Range of values for the parameter being varied.
            initial_exit_thres (float): Fixed value of the other parameter.
            fixed_exit_val (float): Fixed value of the exit_thres
            fixed_entry_val (float): Fixed value of the entry_thres

        Returns:
            raw_prices_fig (fig): (log) raw price plot of two tickers
            kalman_spread_fig (fig): Kalman spread plot of two tickers
            spread_with_signals_fig (fig): Kalman spread with signal labels
            fixed_exit_fig (fig): Sensitivity check 1, with fixed exit_thres
            fixed_entry_fig (fig): Sensitivity check 2, with fixed entry_thres
            results (pd.DataFrame): Dataframe shows the performance metric with initial_entry_thres and initial_exit_thres
            optimized_results (pd.DataFrame): Dataframe shows optimized results with objective function, best entry_thres, best exit_thres, and other performance metrics
        """
        price_pivot_data = DataMixin().pivot_data(DataMixin().load_data(), values='Log Close')

        # Plot raw_prices_fig
        raw_prices_fig = plot_raw_prices(ticker1,ticker2, price_pivot_data, is_log=True)

        # Plot kalman_spread_fig
        kalman_spread, beta_series = self._calculate_spread(ticker1,ticker2, price_pivot_data, method="Kalman")
        kalman_spread_fig = plot_kalman_spread(ticker1,ticker2, kalman_spread, beta_series)

        signals = self.generate_signals_rolling(kalman_spread, initial_entry_thres, initial_exit_thres, window=130, duration_cap=30, min_hold_days=3)
        spread_with_signals_fig = plot_spread_with_signals(ticker1, ticker2, signals, initial_entry_thres, initial_exit_thres)

        results = self.backtest_pair_trading_pipeline(ticker1, ticker2, price_pivot_data, spread_type='Kalman',
                                             entry_thres=initial_entry_thres, exit_thres=initial_exit_thres, window = 130,
                                             transaction_cost=0.0001,
                                             optimize=False,
                                             performance_metrics_list=None)


        optimized_results = self.backtest_pair_trading_pipeline(ticker1, ticker2, price_pivot_data, spread_type='Kalman',
                                             window = 130,
                                             transaction_cost=0.0001,
                                             optimize=True,
                                             performance_metrics_list=['sharpe_ratio',
                                                                        'sortino_ratio',
                                                                        'max_drawdown'])#drawdown_to_vol_ratio

        # Sensitivity: Fixed exit_thres = 1, varied entry
        sharpe = self.calculate_sharpe_ratio(kalman_spread, price_pivot_data, ticker1, ticker2,
                                        varying_values = np.round(np.arange(1.0, 4.1, 0.05), 2),
                                        fixed_value = round(fixed_exit_val, 2),
                                        vary_parameter='entry_thres', criterion='sharpe_ratio')
        fixed_exit_fig = plot_sharpe_ratio(sharpe, vary_parameter = 'entry_thres', fixed_value = round(fixed_exit_val, 2))


        # Fixed entry_thres = 2, varied exit
        sharpe = self.calculate_sharpe_ratio(kalman_spread, price_pivot_data, ticker1, ticker2,
                                        varying_values=np.round(np.arange(0.2, 2.1, 0.05), 2),
                                        fixed_value=round(fixed_entry_val, 2),
                                        vary_parameter='exit_thres', criterion='sharpe_ratio')
        fixed_entry_fig = plot_sharpe_ratio(sharpe, vary_parameter = 'exit_thres', fixed_value = round(fixed_entry_val, 2))

        return raw_prices_fig, kalman_spread_fig, spread_with_signals_fig, fixed_exit_fig, fixed_entry_fig, results, optimized_results


    def backtesting_dashboard(self, ticker1, ticker2, initial_entry_thres=2.0, initial_exit_thres=0.3, fixed_exit_val=1.0,
                               fixed_entry_val=2.0):
        port = 3034
        url = f"http://127.0.0.1:{port}"
        app = dash.Dash(__name__)

        raw_prices_fig, kalman_spread_fig, spread_with_signals_fig, fixed_exit_fig, fixed_entry_fig, results, optimized_results = \
            self.show_board(ticker1, ticker2, initial_entry_thres, initial_exit_thres, fixed_exit_val, fixed_entry_val)

        # Style for tables
        table_style = {
            'width': '100%',
            'borderCollapse': 'collapse',
            'margin': '25px 0',
            'fontSize': '18px',
            'textAlign': 'left'
        }

        th_td_style = {
            'border': '1px solid #ddd',
            'padding': '8px'
        }

        header_style = {
            'backgroundColor': '#f2f2f2',
            'fontWeight': 'bold',
            'textAlign': 'center'
        }

        alternate_row_style = {
            'backgroundColor': '#f9f9f9'
        }

        app.layout = html.Div([
            html.H1("Backtesting Results Dashboard"),

            dcc.Graph(id='raw_prices_fig', figure=raw_prices_fig),
            dcc.Graph(id='kalman_spread_fig', figure=kalman_spread_fig),
            dcc.Graph(id='spread_with_signals_fig', figure=spread_with_signals_fig),

            html.H2(f"Results(with fixed entry threshold={initial_entry_thres} and fixed exit threshod={initial_exit_thres})"),
            html.Table(
                [html.Tr([html.Th(col, style={**th_td_style, **header_style}) for col in results.columns])] +
                [html.Tr([
                    html.Td(results.iloc[i][col], style=th_td_style) for col in results.columns
                ], style=alternate_row_style if i % 2 == 0 else {}) for i in range(len(results))],
                style=table_style
            ),

            html.H2(f"Optimized Results(with different objective function)"),
            html.Table(
                [html.Tr([html.Th(col, style={**th_td_style, **header_style}) for col in optimized_results.columns])] +
                [html.Tr([
                    html.Td(optimized_results.iloc[i][col], style=th_td_style) for col in optimized_results.columns
                ], style=alternate_row_style if i % 2 == 0 else {}) for i in range(len(optimized_results))],
                style=table_style
            ),

            # Adding sensitivity check figures after the tables
            dcc.Graph(id='fixed_exit_fig', figure=fixed_exit_fig),
            dcc.Graph(id='fixed_entry_fig', figure=fixed_entry_fig)
        ])

        
        threading.Timer(1, lambda: webbrowser.open_new(url)).start()
        app.run_server(debug=True, port=port)