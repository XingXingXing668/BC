"""
Created on 12 Nov 2024

@author: Xingyi
"""

import dash.exceptions
import dash_bootstrap_components as dbc
import dash
from dash import dcc, html, Output, Input
import pandas as pd
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go
from src.utils.data_mixin import DataMixin
import threading
import webbrowser

class StockRegressionDashBoard(DataMixin):
    PORT = 3035
    URL = f"http://127.0.0.1:{PORT}"
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.stock_daily_return = self.pivot_data(self.load_raw_daily(), values='Daily Return')
        self.indices_daily_return = self.get_indices_daily_data().pct_change().rename(columns = {"^GSPC": "SP500", "^RUT":"Russel2000", "^NDX":"Nasdaq100"})
        self.app = dash.Dash(__name__)

        self.app.layout = html.Div([
            html.H1("Index & Stock Selector"),

            html.Label("Select Index:"),
            dcc.Dropdown(
                id='index-dropdown',
                options=[{'label': 'S&P 500', 'value': 'SP500'},
                         {'label': 'Russell 2000', 'value': 'Russel2000'},
                         {'label': 'NASDAQ 100', 'value': 'Nasdaq100'}],
                placeholder="Choose an index"
            ),

            html.Label("Select Stocks (Limit: 10):"),
            dcc.Dropdown(
                id='stock-dropdown',
                placeholder="Select stocks",
                multi=True
            ),

            # Warning message showing how many stocks are chosen
            html.Div(id='remaining-count', style={'margin-top': '10px', 'font-weight': 'bold', 'color': 'blue'}),

            # Warning message for exceeding the stock limit
            html.Div(id='warning-message', style={'color': 'red', 'margin-top': '10px'}),

            # Display the regression results directly as formatted text
            html.Div(id='regression-results', style={'margin-top': '20px'}),

            # Graph to display actual vs. predicted returns
            dcc.Graph(id='returns-graph')
        ])

        self._register_callbacks()

    def run_ols_regression(self,selected_df, selected_index, selected_stocks):
        '''
            Function to run OLS regression and return results
        '''
        # Ensure data for at least 252 days after dropping NaNs
        if len(selected_df) < 252:
            return None, None, "Insufficient data: fewer than 252 days."

        # Time range for filtered DataFrame
        time_range = f"Time range: {selected_df.index.min().date()} to {selected_df.index.max().date()}"

        # Regression with the selected index as Y and selected stocks as X
        X = selected_df[selected_stocks]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        Y = selected_df[selected_index]

        # Run OLS regression
        model = sm.OLS(Y, X).fit()
        summary_text = model.summary().as_text()

        # Predicted returns
        predicted_returns = model.predict(X)

        # Construct regression results
        regression_results = html.Div([
            html.Pre(time_range, style={'whiteSpace': 'pre-wrap', 'font-family': 'monospace', 'font-weight': 'bold',
                                        'color': "Purple"}),
            html.Pre(summary_text, style={'whiteSpace': 'pre-wrap', 'font-family': 'monospace'})
        ])
        return predicted_returns, regression_results, None


    def plot_graph(self, selected_index, actual_return, predicted_returns, style='scatter'):
        '''
            Plot actual vs. predicted index performance
        '''
        graph_style = {'scatter': 'markers', 'line': 'line'}
        fig = go.Figure()

        # Add actual index performance as a line and predicted performace as input style.
        fig.add_trace(
            go.Scatter(
                x=predicted_returns.index,
                y=actual_return[selected_index],
                mode='lines',
                name='Actual Index Performance',
                line=dict(width=1.5),
                hovertemplate="Date: %{x}<br>Performance: %{y}"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predicted_returns.index,
                y=predicted_returns,
                mode=graph_style[style], # Can change predited values in line chart
                name='Predicted Index Performance',
                # line=dict(color='red', width=1),
                marker=dict(size=3),
                hovertemplate="Date: %{x}<br>Predicted: %{y}"
            )
        )

        fig.update_layout(
            # template="plotly_dark",
            title="Scatterplot Regression of Index Performance",
            xaxis_title="Date",
            yaxis_title="Performance",
            height=600,  # Stretch vertically for better readability
            hovermode="x unified",
            legend=dict(
                title="Legend",
                x=0.01, y=0.99,
                bgcolor="rgba(0, 0, 0, 0)"
            ),
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(rangeslider_visible=True)

        return fig

    def _register_callbacks(self):
        # Function to update stock dropdown based on selected index
        @self.app.callback(
            Output('stock-dropdown', 'options'),
            Input('index-dropdown', 'value')
        )
        def update_stock_dropdown(selected_index):
            if selected_index == "SP500":
                tickers = self.get_sp500_tickers()
            elif selected_index == "Russel2000":
                tickers = self.get_russel200_tickers()
            elif selected_index == "Nasdaq100":
                tickers = self.get_ndq100_tickers()
            else:
                tickers = []

            # Return options for stock dropdown
            return [{'label': ticker, 'value': ticker} for ticker in tickers]

        @self.app.callback(
            [Output('remaining-count', 'children'),
             Output('warning-message', 'children'),
             Output('regression-results', 'children'),
             Output('returns-graph', 'figure')],
            [Input('stock-dropdown', 'value'), Input('index-dropdown', 'value')]
        )
        def update_remaining_count(selected_stocks, selected_index):
            remaining_message = ""
            warning_message = ""
            regression_results = ""
            figure = go.Figure()  # Will return empty figure for cases where the conditions aren't met

            # Calculate remaining stocks needed
            if selected_stocks:
                remaining_stocks = 10 - len(selected_stocks)
                remaining_message = f"Stocks selected: {len(selected_stocks)} out of 10. Remaining: {remaining_stocks}"
                if remaining_stocks < 0:
                    warning_message = "You have selected more than 10 stocks. Please select only 10."
                    return remaining_message, warning_message, regression_results, figure
            else:
                remaining_message = "Stocks selected: 0 out of 10. Remaining: 10"

            # Run check only if exactly 10 stocks are selected
            if selected_stocks and len(selected_stocks) == 10:
                # Keep only the selected index and stocks in the DataFrame
                stock_raw_daily_return = self.pivot_data(self.load_raw_daily(), values='Daily Return')
                indices_daily_return = self.get_indices_daily_data().pct_change().rename(columns = {"^GSPC": "SP500", "^RUT":"Russel2000", "^NDX":"Nasdaq100"})
                df = self._merged_data(indices_daily_return, stock_raw_daily_return)
                selected_df = df[[selected_index] + selected_stocks]

                # Identify tickers with too many missing values
                initial_row_count = len(selected_df)
                problematic_tickers = [
                    ticker for ticker in selected_stocks
                    if initial_row_count - selected_df[ticker].isna().sum() < 252
                ]

                # If any problematic tickers, show warning
                if problematic_tickers:
                    problematic_tickers_message = ", ".join(problematic_tickers)
                    warning_message = html.Div([
                        "Some stocks in the selected tickers set contain too many missing values, resulting in fewer than 252 days of data.",
                        html.Br(),
                        f"Problematic tickers: {problematic_tickers_message}. Please replace these stocks."
                    ])
                    return remaining_message, warning_message, regression_results, figure

                # Drop NaNs and run OLS regression
                selected_df = selected_df.dropna()
                predicted_returns, regression_results, error_message = self.run_ols_regression(selected_df, selected_index,
                                                                                          selected_stocks)

                # In case of there was an error (e.g., not enough rows)
                if error_message:
                    warning_message = error_message
                    return remaining_message, warning_message, regression_results, figure

                figure = self.plot_graph(selected_index, selected_df, predicted_returns)
            return remaining_message, warning_message, regression_results, figure


    def run_app(self):
        threading.Timer(1, lambda: webbrowser.open_new(self.URL)).start()
        self.app.run_server(debug=True, use_reloader=False, threaded=True, port=self.PORT)