"""
Created on 12 Nov 2024

@author: Xingyi
"""

from src.pair_trade import PairTradeAnalysis
from src.pair_trading_dash_board import PairTradingDashBoard
from src.backtesting import BackTesting
from src.utils.plot_utils import plot_regression_results
from src.stock_regression_dash_board import StockRegressionDashBoard
from src.most_explain_dash_board import  MostExplainDashBoard


def Project_1_DashBoard(): # http://127.0.0.1:3033
    # Project1 step 1/2: Pair Trading Strategy
    PairTradingDashBoard(PairTradeAnalysis()).run_app() 
    
def Project_1_BackTest(ticker1='WVE', ticker2='XERS'): #http://127.0.0.1:3034
    bt = BackTesting()
    bt.backtesting_dashboard(ticker1, ticker2)  # This would start the Dash server and display the webpage with plots and tables.

def Project2(): 
    # Project2: Regression Analysis
    StockRegressionDashBoard().run_app() # http://127.0.0.1:3035
    MostExplainDashBoard().run_app()# http://127.0.0.1:3036
    
if __name__ == "__main__":
    ########################################################################################
    # Note: Due to GitHub's 25MB file size limit, I couldn't upload the full data. 
    # Instead, a text file in the "Data" folder provides a Dropbox link to access the full data to run the code.
    #######################################################################################
    
    # Pop-up a dashboard displaying highly correlated pairs (approximately 3 minutes to run with multi-threading)
    Project_1_DashBoard()

    # If you want to try other ticker pairs, you can run this line and outcomment previous step
    Project_1_BackTest('ALLK','BCAB')

    # Run project 2
    Project2()
    
