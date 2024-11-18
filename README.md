This folder contains implementations for the following two projects, with detailed procedures documented in the Demo file.

--------------------------------------------------------------------------------------------------
In order to run the project, user can choose either

(1) run the jupyter notebook "Demo.ipynb", which show the detailed procedure and results.<br>
(2) run the main.py file, it will pop up four web browsers with port addresses above.<br>

Note: Due to GitHub's 25MB file size limit, I couldn't upload the full data. Instead, a text file in the "Data" folder provides a Dropbox link to access the full data. To run the code, please download the data from Dropbox link, and make sure the downloaded data in the "Data" foler.
  
--------------------------------------------------------------------------------------------------

# Project 1: Pair Trading

### Part 1: Identifying Highly Correlated Pairs
A dashboard (http://127.0.0.1:3033/) will automatically pop-up and allow users to select top pairs based on OLS spread and Kalman spread.
Please note that this process takes approximately 3 minutes with multi-threading. If you run it in the Jupyter notebook "Demo.ipynb," a progress bar will display progress at the bottom of the output cell.
Please see "Project_1_HighCorrelation.pdf" for a quick review of results.


### Part 2: Backtesting Selected Pairs
Backtesting features are included to evaluate performance metrics across different entry/exit thresholds, along with a sensitivity analysis. This generates a Plotly HTML report. (http://127.0.0.1:3034/)
Please see "Project_1_BackTest.pdf" for a quick review of results.


# Project 2: Multi-Variate Index
### Main part: Index Regression
A dashboard (http://127.0.0.1:3035/) enables users to choose an index, set the start and end dates, and select up to ten tickers for regression analysis.
Please see "Project_2_Regression.pdf" for a quick review of results.


### Optional part: Top Stock Selection for Index Explanation
A dashboard (http://127.0.0.1:3036/)  enable users to select an index, set a date range, and choose a method to find the top 10 stocks explaining index performance through a dedicated dashboard.
Please see "Project_2_MostExplain.pdf" for a quick review of results.

--------------------------------------------------------------------------------------------------



