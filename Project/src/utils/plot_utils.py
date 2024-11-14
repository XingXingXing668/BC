import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pykalman import KalmanFilter


def plot_raw_prices(ticker1, ticker2, pivot_prices, is_log=False):
    """
    Create a figure for the price series of two tickers, with a title indicating
    whether raw prices or log-transformed prices are used.
    Parameters:
        ticker1 (str): The first ticker symbol.
        ticker2 (str): The second ticker symbol.
        pivot_prices (pd.DataFrame): Pivoted DataFrame with prices, indexed by date, and tickers as columns.
        is_log (bool): Set to True if log prices are being plotted. Defaults to False.
    Returns:
        fig (go.Figure): Figure object ready to be used in Dash.
    """
    # Check if both tickers are in the DataFrame
    if ticker1 in pivot_prices.columns and ticker2 in pivot_prices.columns:
        # Determine the y-axis label and title based on log setting
        price_type = 'Log(Closing Price)' if is_log else 'Closing Price'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pivot_prices.index,
            y=pivot_prices[ticker1],
            mode='lines',
            name=ticker1,
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=pivot_prices.index,
            y=pivot_prices[ticker2],
            mode='lines',
            name=ticker2,
            line=dict(color='red')
        ))

        # Update layout for title and labels
        fig.update_layout(
            title=f'{price_type} Series of {ticker1} and {ticker2}',
            xaxis_title='Date',
            yaxis_title=price_type,
            legend_title="Tickers"
        )
        return fig
    else:
        raise ValueError("One or both tickers are not found in the pivot_prices DataFrame.")


def plot_ols_spread(ticker1, ticker2, spread, beta):
    """
    Create a Plotly figure for OLS spread analysis, including mean and standard deviation bands.
    Parameters:
        ticker1 (str): The first ticker symbol.
        ticker2 (str): The second ticker symbol.
        spread (pd.Series): The spread between the two tickers over time.
        beta (float): The OLS beta coefficient.
    Returns:
        fig (go.Figure): Figure object ready to be used in Dash.
    """
    mean_spread = spread.mean()
    std_spread = spread.std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread.values,
        mode='lines',
        name=f'Spread ({ticker1} - {beta:.3f} * {ticker2})',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread] * len(spread),
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread + std_spread] * len(spread),
        mode='lines',
        name='+1 Std Dev',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread - std_spread] * len(spread),
        mode='lines',
        name='-1 Std Dev',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread + 2 * std_spread] * len(spread),
        mode='lines',
        name='+2 Std Dev',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread - 2 * std_spread] * len(spread),
        mode='lines',
        name='-2 Std Dev',
        line=dict(color='orange', dash='dash')
    ))

    # Update layout for title and labels
    fig.update_layout(
        title=f'OLS Spread Analysis: {ticker1} and {ticker2}',
        xaxis_title='Date',
        yaxis_title='Spread',
        legend_title="Legend"
    )
    return fig

def plot_kalman_spread(ticker1, ticker2, spread, beta_series):
    """
    Create a figure for Kalman filter spread analysis, including mean and standard deviation bands.
    Parameters:
        ticker1 (str): The first ticker symbol.
        ticker2 (str): The second ticker symbol.
        spread (pd.Series): The spread between the two tickers over time.
        beta_series (pd.Series): The series of beta coefficients from the Kalman filter.
    Returns:
        fig (go.Figure): Figure object ready to be used in Dash.
    """
    mean_spread = spread.mean()
    std_spread = spread.std()
    avg_beta = beta_series.mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread.values,
        mode='lines',
        name=f'Kalman Filter Spread ({ticker1} - {avg_beta:.2f} * {ticker2})',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread] * len(spread),
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread + std_spread] * len(spread),
        mode='lines',
        name='+1 Std Dev',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread - std_spread] * len(spread),
        mode='lines',
        name='-1 Std Dev',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread + 2 * std_spread] * len(spread),
        mode='lines',
        name='+2 Std Dev',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[mean_spread - 2 * std_spread] * len(spread),
        mode='lines',
        name='-2 Std Dev',
        line=dict(color='orange', dash='dash')
    ))

    # Update layout for title and labels
    fig.update_layout(
        title=f'Kalman Filter Spread Analysis: {ticker1} and {ticker2}',
        xaxis_title='Date',
        yaxis_title='Spread',
        legend_title="Legend"
    )
    return fig


def plot_spread_with_signals(ticker1, ticker2, signals, entry_k, exit_k):
    """
    Create a Plotly figure to plot the spread with entry and exit signals overlaid.

    Parameters:
        ticker1 (str): name of the first ticker in the pair.
        ticker2 (str): name of the second ticker in the pair.
        signals (str): pd.DataFrame containing 'spread' and 'trade' columns with trading signals.
        entry_k (float): the entry threshold as a multiplier of the rolling standard deviation.
        exit_k (float) :the exit threshold as a multiplier of the rolling standard deviation.
    Returns:
        fig (go.Figure): figure object ready to be used in Dash.
    """
    spread = signals['spread']

    # Identify entry and exit
    entry_long_points = signals[signals['trade'] == 1].index  # Entry long
    exit_long_points = signals[signals['trade'] == 2].index  # Exit long
    entry_short_points = signals[signals['trade'] == -1].index  # Entry short
    exit_short_points = signals[signals['trade'] == -2].index  # Exit short

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread.values,
        mode='lines',
        name=f'Spread ({ticker1} - {ticker2})',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=entry_long_points,
        y=spread.loc[entry_long_points],
        mode='markers',
        name='Entry Long',
        marker=dict(symbol='triangle-up', color='purple', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=exit_long_points,
        y=spread.loc[exit_long_points],
        mode='markers',
        name='Exit Long',
        marker=dict(symbol='circle', color='green', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=entry_short_points,
        y=spread.loc[entry_short_points],
        mode='markers',
        name='Entry Short',
        marker=dict(symbol='triangle-down', color='brown', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=exit_short_points,
        y=spread.loc[exit_short_points],
        mode='markers',
        name='Exit Short',
        marker=dict(symbol='x', color='red', size=10)
    ))

    fig.update_layout(
        title=f'Spread Analysis with Trade Signals: {ticker1} and {ticker2} <br> Entry k: {entry_k}, Exit k: {exit_k}',
        xaxis_title='Date',
        yaxis_title='Spread',
        legend_title="Legend",
        showlegend=True,
        template='plotly_white'
    )

    return fig


def plot_cumulative_return(signals_return):
    """
    Create figure to plot the cumulative return over time from the signals DataFrame.
    Parameters:
        signals_return (pd.DataFrame): containing a 'cumulative_return' column.
    Returns:
        fig (go.Figure): Figure object ready to be used in Dash.
    """
    # Check for 'cumulative_return' column
    if 'cumulative_return' not in signals_return.columns:
        raise ValueError("signals DataFrame must contain a 'cumulative_return' column.")

    start_point = (signals_return.index[0], signals_return['cumulative_return'].iloc[0])
    end_point = (signals_return.index[-1], signals_return['cumulative_return'].iloc[-1])
    max_point = (signals_return['cumulative_return'].idxmax(), signals_return['cumulative_return'].max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=signals_return.index,
        y=signals_return['cumulative_return'],
        mode='lines',
        name='Cumulative Return',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=[start_point[0]], y=[start_point[1]],
        mode='markers', name='Start',
        marker=dict(color='green', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[end_point[0]], y=[end_point[1]],
        mode='markers', name='End',
        marker=dict(color='red', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[max_point[0]], y=[max_point[1]],
        mode='markers', name='Max Return',
        marker=dict(color='orange', size=10)
    ))

    fig.update_layout(
        title='Cumulative Return Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend_title="Legend",
        template='plotly_white'
    )

    return fig

def plot_sharpe_ratio(sharpe_data, vary_parameter, fixed_value):
    """
    Create a figure to plot the Sharpe Ratio against either Entry_k or Exit_k, with the other parameter fixed.
    Parameters:
        sharpe_data (pd.DataFrame): DataFrame containing the varying parameter and 'Sharpe Ratio' columns.
        vary_parameter (str): The parameter being varied ('entry_k' or 'exit_k').
        fixed_value (float): The fixed value of the other parameter.
    Returns:
        fig (go.Figure): figure object.
    """
    # Determine axis titles and plot title based on the varied parameter
    xaxis_title = 'Entry_thres' if vary_parameter == 'entry_k' else 'Exit_thres'
    title = f'Sharpe Ratio by varying {xaxis_title} (with fixed {"Exit_thres" if vary_parameter == "entry_k" else "Entry_thres"} = {fixed_value:.2f})'

    # Find the maximum Sharpe Ratio and the corresponding parameter value
    max_sharpe = sharpe_data['Sharpe Ratio'].max()
    optimal_value = sharpe_data.loc[sharpe_data['Sharpe Ratio'].idxmax(), vary_parameter]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sharpe_data[vary_parameter],
        y=sharpe_data['Sharpe Ratio'],
        mode='lines+markers',
        name='Sharpe Ratio',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=[optimal_value],
        y=[max_sharpe],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'Optimal {xaxis_title}: {optimal_value:.2f}, Max Sharpe: {max_sharpe:.2f}'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title='Sharpe Ratio',
        width=1000,
        height=600,
        font=dict(size=14),
        showlegend=True,
        legend=dict(
            x=0.8, y=0.95,  # Position legend in the top-right corner
            xanchor='center', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='Black', borderwidth=1
        ),
        template='plotly_white'
    )

    return fig


def plot_regression_results(daily_changes, style='scatter'):
    graph_style = {'scatter':'markers', 'line': 'line'}
    index_name = daily_changes['index_name'][0]
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_changes['Date'],
            y=daily_changes[index_name],
            mode='lines',
            name='Actual Index Performance',
            line=dict( width=1.5),
            hovertemplate="Date: %{x}<br>Performance: %{y}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_changes['Date'],
            y=daily_changes['predicted_index'],
            mode=graph_style[style],
            # mode='lines',
            name='Predicted Index Performance',
            # line=dict(color='red', width=1),
            marker=dict( size=2.5),  # Smaller dot size
            hovertemplate="Date: %{x}<br>Predicted: %{y}"
        )
    )

    fig.update_layout(  
        template="plotly_dark", 
        title="Scatterplot Regression of Index Performance",
        xaxis_title="Date",
        yaxis_title="Performance",
        height=600,  # Stretch vertically for better readability
        hovermode="x unified",
        legend=dict(
            title="Legend",
            x=0.01, y=0.99,
            # bgcolor="rgba(255, 255, 255, 0.5)"
        ),
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(rangeslider_visible=True)
    fig.show(renderer="browser")

def plot_betas(top_stocks, betas, method="Lasso", index="S&P 500"):
    """
    Plots beta estimates for the top selected stocks over time.
    Parameters:
        betas (pd.DataFrame/pd.Series): DataFrame of time-varying beta estimates or a Series of static beta.
        top_stocks (pd.Index/list): List of the top stocks to plot.
        method (str): Method used to calculate betas. Determines whether plot is dynamic or static.
        index (str): The name of the selected index.
    Returns:
        fig (go.Figure): figure object.
    """
    top_stocks = list(top_stocks)
    dynamic = method == "Kalman"

    fig = go.Figure()

    if dynamic:
        # Plot dynamic betas (time-varying)
        top_dynamic_betas = betas[top_stocks]
        for stock in top_stocks:
            fig.add_trace(go.Scatter(
                x=top_dynamic_betas.index,
                y=top_dynamic_betas[stock],
                mode='lines',
                name=stock
            ))
        fig.update_layout(
            title=f"Dynamic Betas of Top Selected Stocks Over Time\n(Index: {index}, Method: {method})",
            xaxis_title="Date",
            yaxis_title="Beta",
            legend_title="Stocks",
            template="plotly_white",
            width=900,
            height=450
        )
    else:
        # Plot static betas (single values)
        fig.add_trace(go.Bar(
            x=top_stocks,
            y=betas[top_stocks],
            text=betas[top_stocks].round(2),
            textposition='outside',
            name="Static Betas"
        ))
        fig.update_layout(
            title=f"Static Betas of Top Selected Stocks\n(Index: {index}, Method: {method})",
            xaxis_title="Stocks",
            yaxis_title="Beta",
            template="plotly_white",
            width=900,
            height=500
        )

    return fig