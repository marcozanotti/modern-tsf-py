# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 3: Nixtla & Sktime ----------------------------------------------
# Marco Zanotti

# Goals:
# - Learn the Nixtla Workflow
# - Learn the Sktime Workflow
# - Understand Accuracy Measurements
# - Understand the Forecast Horizon & Confidence Intervals
# - Understand refitting

# Challenges:
# - Challenge - Nixtla
# - Challenge - Sktime



# Packages ----------------------------------------------------------------

import numpy as np
import pandas as pd
# import polars as pl
import pickle

import pytimetk as tk

pd.set_option("display.max_rows", 5)



# Data --------------------------------------------------------------------

# Pandas
subscribers_df = pd.read_csv(
    "data/subscribers/subscribers.csv", 
    parse_dates=['optin_time']
)
analytics_df = pd.read_csv(
    "data/subscribers/analytics_hourly.csv", 
    parse_dates=['dateHour'],
    date_format='%Y%m%d%H'
)
events_df = pd.read_csv(
    "data/subscribers/events.csv", parse_dates=['event_date']
)

subscribers_df.glimpse()
subscribers_df
analytics_df.glimpse()
events_df.glimpse()


with open('artifacts/Python/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)




