"""
Created on 12 Nov 2024

@author: Xingyi
"""

import threading
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import pandas as pd
from tqdm import tqdm
import threading
import webbrowser

class PairTradingDashBoard:
    PORT = 3033
    URL = f"http://127.0.0.1:{PORT}"
    def __init__(self, analysis_class):
        self.analysis_class = analysis_class
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Define the global progress state
        self.progress_state = {'value': 0, 'completed': False}
        self.analysis_results = {'OLS': None, 'Kalman': None}

        self.app.layout = html.Div([
            html.H1("Pairs Trading Analysis Dashboard (Find Highly Correlated Pairs)",
                    style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '5px'}),
            html.Label("Select Start Date:"),
            dcc.DatePickerSingle(
                id='start-date',
                min_date_allowed='2015-01-01',
                max_date_allowed='2024-11-12',
                initial_visible_month='2015-01-01',
                date='2015-01-01'
            ),
            html.Label("Select End Date:"),
            dcc.DatePickerSingle(
                id='end-date',
                min_date_allowed='2015-01-01',
                max_date_allowed='2024-11-12',
                initial_visible_month='2024-11-12',
                date='2024-11-12'
            ),
            html.Br(),
            dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, style={"height": "30px"}),
            html.Br(),
            html.Button("Start Analysis", id="start-button", n_clicks=0),
            html.Br(),
            dcc.Interval(id='interval-progress', interval=1000, n_intervals=0, disabled=True),
            html.H3("OLS Analysis Results (Top 10 Stationary Spread by Vol)"),
            dcc.Loading(
                id="loading-ols",
                type="circle",
                children=html.Div(id='output-table-ols')
            ),
            html.Br(),
            html.H3("Kalman Filter Analysis Results (Top 10 Stationary Spread by Vol)"),
            dcc.Loading(
                id="loading-kalman",
                type="circle",
                children=html.Div(id='output-table-kalman')
            ),
        ])

        self._register_callbacks()

    OUTPUT_CALLBACKS = [
                Output('interval-progress', 'disabled'),
                Output('progress-bar', 'value'),
                Output('output-table-ols', 'children'),
                Output('output-table-kalman', 'children')
            ]
    INPUT_CALLBACKS = [Input('start-button', 'n_clicks'), Input('interval-progress', 'n_intervals')]
    STATES = [State('start-date', 'date'), State('end-date', 'date')]

    def _register_callbacks(self):
        @self.app.callback(self.OUTPUT_CALLBACKS, self.INPUT_CALLBACKS, self.STATES)
        def manage_analysis_and_progress(n_clicks, n_intervals, start_date, end_date):
            triggered_id = ctx.triggered_id

            if triggered_id == 'start-button' and n_clicks > 0:
                thread = threading.Thread(target=self.run_analysis_in_thread, args=(start_date, end_date))
                thread.start()
                return False, 0, dash.no_update, dash.no_update  # Enable interval, reset progress bar

            if triggered_id == 'interval-progress':
                progress_value = self.progress_state['value']
                if self.progress_state['completed']:
                    if isinstance(self.analysis_results['OLS'], str):
                        return True, progress_value, html.Div(self.analysis_results['OLS'], style={'color': 'red', 'font-weight': 'bold'}), html.Div(self.analysis_results['Kalman'], style={'color': 'red', 'font-weight': 'bold'})
                    else:
                        ols_table = dbc.Table.from_dataframe(
                            self.analysis_results['OLS'],
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True
                        )
                        kalman_table = dbc.Table.from_dataframe(
                            self.analysis_results['Kalman'],
                            striped=True,
                            bordered=True,
                            hover=True,
                            responsive=True
                        )
                        return True, progress_value, ols_table, kalman_table

                return False, progress_value, dash.no_update, dash.no_update

            return True, 0, dash.no_update, dash.no_update

    def run_analysis_in_thread(self, start_date, end_date):
        try:
            self.progress_state['value'] = 0
            self.progress_state['completed'] = False
            print("Analysis started")

            # Run the actual workflow from the analysis class
            results = self.analysis_class.run_pairs_trading_workflow(self.analysis_class.load_data(), start_date, end_date)

            if isinstance(results, str):
                self.analysis_results['OLS'] = results
                self.analysis_results['Kalman'] = results
            else:
                self.analysis_results['OLS'] = results['OLS'].head(10)
                self.analysis_results['Kalman'] = results['Kalman'].head(10)

            self.progress_state['value'] = 100
            self.progress_state['completed'] = True
            print("Analysis completed successfully")
        except Exception as e:
            print(f"Error in analysis: {e}")

    def run_app(self):
        threading.Timer(1, lambda: webbrowser.open_new(self.URL)).start()
        self.app.run_server(debug=True, use_reloader=False, threaded=True, port=self.PORT)
        



