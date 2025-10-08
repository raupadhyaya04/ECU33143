# Need to fill the blanks in the crypto and macro data

import pandas as pd

macro_df = pd.read_csv("./Macro Data/macro_fred_data.csv", index_col=0, parse_dates=True)
yahoo_df = pd.read_csv("./Macro Data/macro_yahoo.csv", index_col=0, parse_dates=True)
macro_df = macro_df.ffill()
yahoo_df = yahoo_df.ffill()
print(macro_df.head())
print(yahoo_df.head())

macro_df.to_csv("./Macro Data/macro_fred_data_filled.csv")
yahoo_df.to_csv("./Macro Data/macro_yahoo_data_filled.csv")