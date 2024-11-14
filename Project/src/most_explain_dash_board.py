"""
Created on 12 Nov 2024

@author: Xingyi
"""

import dash
from dash import Dash, dcc, html, dash_table, Output, Input
import plotly.graph_objects as go
import threading
import webbrowser

from src.most_explain import MostExplainStocks

class MostExplainDashBoard(MostExplainStocks):
    PORT = 3036
    URL = f"http://127.0.0.1:{PORT}"
    def __init__(self):
        self.app = Dash(__name__)

        self.app.layout = html.Div([
            html.H1("Top 10 Stocks Explaining the Chosen Index"),

            dcc.Dropdown(
                id="index-dropdown",
                options=[
                    {"label": "S&P500", "value": "SP500"},
                    {"label": "Russel2000", "value": "Russel2000"},
                    {"label": "Nasdaq100", "value": "Nasdaq100"}
                ],
                value="SP500", 
                placeholder="Select an index"
            ),

            dcc.Dropdown(
                id="method-dropdown",
                options=[
                    {"label": "Stepwise", "value": "Stepwise"},
                    {"label": "Lasso", "value": "Lasso"},
                    {"label": "Kalman", "value": "Kalman"}
                ],
                value="Kalman", 
                placeholder="Select a method"
            ),

            # Date range
            dcc.DatePickerRange(
                id="date-picker-range",
                start_date="2015-01-01",
                end_date="2024-11-12"
            ),

            # Button to trigger analysis
            html.Button("Find Top Stocks", id="analyze-button", n_clicks=0),

            # Output area for plots
            dcc.Graph(id="beta-plot")
        ])
        self._register_callbacks()

    def _register_callbacks(self):
        # Callback to update the plot
        @self.app.callback(
            Output("beta-plot", "figure"),
            Input("analyze-button", "n_clicks"),
            Input("index-dropdown", "value"),
            Input("method-dropdown", "value"),
            Input("date-picker-range", "start_date"),
            Input("date-picker-range", "end_date")
        )
        def update_plot(n_clicks, chosen_index, method, start_date, end_date):
            if n_clicks > 0:
                # Run the find_most_explain_stocks function with the selected inputs
                daily_stock_return = self.pivot_data(self.load_raw_daily(), values='Daily Return')
                daily_index_return = self.get_indices_daily_data().pct_change()
                daily_index_return = daily_index_return.rename(
                    columns={"^GSPC": "SP500", "^RUT": "Russel2000", "^NDX": "Nasdaq100"})

                fig = self.find_most_explain_stocks(
                    daily_stock_return,
                    daily_index_return,
                    chosen_index=chosen_index,
                    start_date=start_date,
                    end_date=end_date,
                    method=method
                )
                return fig
            return go.Figure()  # Return an empty figure before the first click

    def run_app(self):
        threading.Timer(1, lambda: webbrowser.open_new(self.URL)).start()
        self.app.run_server(debug=True, use_reloader=False, threaded=True, port=self.PORT)