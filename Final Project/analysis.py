"""
Plan rn:
1) Get the data linked via multi-regression
2) Perform regression analysis
3) Visualize the results
4) Move to the next stage
"""
import pandas as pd

macro_fred = pd.read_csv("./Macro Data/macro_fred_data.csv", index_col=0, parse_dates=True)
macro_yahoo = pd.read_csv("./Macro Data/macro_yahoo.csv", index_col=0, parse_dates=True)