# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 1: Manipulation, Transformation & Visualization -----------------
# Marco Zanotti

# Goals:
# - Learn timetk data wrangling functionality
# - Commonly used time series transformations
# - Commonly used time series visualizations


# Packages ----------------------------------------------------------------

import numpy as np
import pandas as pd
# import polars as pl
import pytimetk as tk
from scipy.stats import boxcox 
from statsmodels.nonparametric.smoothers_lowess import lowess

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
analytics_df.glimpse()
events_df.glimpse()

# Polars
# subscribers_pl_df = pl.read_csv(
#     "data/subscribers/subscribers.csv", try_parse_dates=True
# )
# analytics_pl_df = pl.read_csv(
#     "data/subscribers/analytics_hourly.csv", try_parse_dates=True
# )
# events_pl_df = pl.read_csv(
#     "data/subscribers/events.csv", try_parse_dates=True
# )
# subscribers_pl_df.glimpse()



# Manipulation ------------------------------------------------------------

# * Summarize by Time -----------------------------------------------------

# - Apply commonly used aggregations
# - High-to-Low Frequency

# to daily
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    )

subscribers_df \
    .assign(value = 1) \
    .groupby(['member_rating']) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    )

# to weekly
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'W',
        agg_func = 'sum'
    )

# to monthly
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'M',
        agg_func = 'sum'
    )

# other aggregations
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'M',
        agg_func = ['mean', 'std', 'median']
    )

subscribers_daily_df = subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    )

analytics_daily_df = analytics_df \
    .rename(columns={'dateHour' : 'date'}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = ['pageViews', 'organicSearches', 'sessions'],
        freq = 'D',
        agg_func = 'sum'
    )


# * Pad by Time -----------------------------------------------------------

# - Filling in time series gaps
# - Low-to-High Frequency (un-aggregating)

# fill daily gaps
subscribers_daily_df \
    .pad_by_time(
        date_column = 'optin_time',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0)
    

# * Filter by Time --------------------------------------------------------

# - Pare data down before modeling
subscribers_daily_df \
    .filter_by_time(
        date_column = 'optin_time',
        start_date = '2018-11-20'
    )

subscribers_daily_df \
    .filter_by_time(
        date_column = 'optin_time',
        start_date = '2019',
        end_date = '2020'
    )


# * Apply by Time ---------------------------------------------------------

# - Get change from beginning/end of period

# first, last, mean, median by period
subscribers_daily_df \
    .apply_by_time(
        date_column = 'optin_time',
        freq = '2W',
        value_mean = lambda df: df['value'].mean(),
        value_median = lambda df: df['value'].median(),
        value_max = lambda df: df['value'].max(),
        value_min = lambda df: df['value'].min()
    )


# * Join by Time ----------------------------------------------------------

# - Investigating Relationships
# - Identify External Regressors

# left join
subscribers_google_joined_df = subscribers_daily_df \
    .pad_by_time(
        date_column = 'optin_time',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0) \
    .merge(
        analytics_daily_df, 
        left_on = 'optin_time', 
        right_on = 'date',
        how = 'left'
    ) \
    .drop('date', axis = 1) \
    .fillna(0)

# long format
subscribers_joined_long_df = subscribers_google_joined_df \
    .melt(
        id_vars = 'optin_time', 
        value_vars = ['value', 'pageViews', 'organicSearches', 'sessions'],
        value_name = 'value_ts'
    )
    
# plot relationships  
subscribers_joined_long_df \
    .query('variable == "value"') \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts'
    )

subscribers_joined_long_df \
    .groupby('variable') \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        facet_ncol = 2, 
        facet_nrow = 2,
        facet_scales = "free_y",
        smooth = False
    )

subscribers_joined_long_df \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )


# * Future Frame ----------------------------------------------------------

# - Forecasting helper
subscribers_daily_df = subscribers_daily_df \
    .pad_by_time(
        date_column = 'optin_time',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0)

subscribers_daily_df \
    .future_frame(date_column = 'optin_time', length_out = 60)

future_df = subscribers_daily_df \
    .future_frame(
        date_column = 'optin_time', 
        length_out = 60, 
        bind_data = False
    )
future_df

# modelling example on date features



# Transformation ----------------------------------------------------------

subscribers_daily_df = subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'optin_time',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0)
subscribers_daily_df['value'] = subscribers_daily_df['value'].astype('int64')

# * Variance Reduction ----------------------------------------------------

# Log
subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log)

# Log + 1
subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p)

# - inversion with np.exp() and np.expm1()

# Box-Cox
def boxcox_vec(x):
    return boxcox(x + 1)[0]

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = boxcox_vec)

# - inversion with the lambda value


# * Range Reduction -------------------------------------------------------

# - Used in visualization to overlay series
# - Used in ML for models that are affected by feature magnitude (e.g. linear regression)

# Normalization Range (0,1)
def min_max(x):
    return (x - min(x)) / (max(x) - min(x))

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = min_max)

# Standardization
def normalize(x):
    return (x - np.mean(x)) / np.std(x)

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = normalize)


# * Smoothing -------------------------------------------------------------

# - Identify trends and cycles
# - Clean seasonality
lowess(
    subscribers_daily_df['value'], 
    range(len(subscribers_daily_df['value'])),
    frac = 0.1
)

def smoother(x, frac = 0.1):
    res = lowess(x, range(len(x)), frac)[:, 1]
    return res

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = smoother)

subscribers_daily_df \
    .transform_columns(
        columns = 'value', 
        transform_func = lambda x: smoother(x, frac = 0.3)
)


# * Rolling Averages ------------------------------------------------------

# - Common time series operations to visualize trend
# - A simple transformation that can help create improve features
# - Can help with outlier-effect reduction & trend detection
# - Note: Businesses often use a rolling average as a forecasting technique
# A rolling average forecast is usually sub-optimal (good opportunity for you!).

subscribers_daily_df['value'].rolling(7).sum()
subscribers_daily_df['value'].rolling(7).mean()

subscribers_daily_df \
    .augment_rolling(
        date_column = 'optin_time',
        value_column = 'value',
        window_func = 'mean',
        window = 7
    )

subscribers_daily_df \
    .augment_rolling(
        date_column = 'optin_time',
        value_column = 'value',
        window_func = ['mean', "std", "sum"],
        window = [7, 14, 30],
        center = True
    )

subscribers_daily_df \
    .augment_rolling(
        date_column = 'optin_time',
        value_column = 'value',
        window_func = ['mean'],
        window = [7, 14, 30]
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_rolling(
        date_column = 'optin_time',
        value_column = 'value',
        window_func = ['mean'],
        window = [7, 14, 30, 90, 365]
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )

# Exponential Weighted Moving Average
subscribers_daily_df['value'].ewm(alpha = 0.1).mean()

subscribers_daily_df \
    .augment_ewm(
        date_column = 'optin_time',
        value_column = 'value',
        window_func = 'mean',
        alpha = 0.1
    )

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_ewm(
        date_column = 'optin_time',
        value_column = 'value',
        window_func = 'mean',
        alpha = 0.1
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )


# * Missing Values Imputation ---------------------------------------------

# - Imputation helps with filling gaps (if needed)

# pd.DataFrame.fillna # fill with a value
# pd.DataFrame.ffill # forward fill
# pd.DataFrame.bfill # backward fill
# pd.DataFrame.interpolate # interpolation


# * Anomaly Cleaning ------------------------------------------------------

# - Outlier removal helps linear regression detect trend and reduces high leverage points
# WARNING: Make sure you check outliers against events
# - usually there is a reason for large values

# Anomaly detection

subscribers_daily_df \
    .anomalize(
        date_column = 'optin_time', value_column = 'value',
        method = 'twitter',
        iqr_alpha = 0.1,
        max_anomalies = 0.2,
        clean_alpha = 0.75,
        clean = 'min-max'
    )

subscribers_anomaly_df = subscribers_daily_df \
    .anomalize(
        date_column = 'optin_time', value_column = 'value',
        method = 'twitter',
        iqr_alpha = 0.05,
        max_anomalies = 0.2,
        clean_alpha = 0.75,
        clean = 'min-max'
    )
subscribers_anomaly_df \
    .plot_anomalies(date_column = 'optin_time')
subscribers_anomaly_df \
    .plot_anomalies_decomp(date_column = 'optin_time')
subscribers_anomaly_df \
    .plot_anomalies_cleaned(date_column = 'optin_time')

# effects on time series regression
# without log
# outlier effect - before cleaning

# outlier effect - after cleaning

# with log
# outlier effect - before cleaning

# outlier effect - after cleaning


# * Lags & Differencing ---------------------------------------------------

# - Lags: Often used for feature engineering
# - Lags: Autocorrelation
# - MOST IMPORTANT: Can possibly use lagged variables in a model, if lags are correlated
# - Difference: Used to go from growth to change
# - Difference: Makes a series "stationary" (potentially)

# lags
subscribers_daily_df \
    .augment_lags(
        date_column = 'optin_time',
        value_column = 'value',
        lags = 1
    )

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_lags(
        date_column = 'optin_time',
        value_column = 'value',
        lags = [1, 7, 14, 30, 90]
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )

# differencing
subscribers_daily_df \
    .augment_diffs(
        date_column = 'optin_time',
        value_column = 'value',
        periods = 1
    )

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_diffs(
        date_column = 'optin_time',
        value_column = 'value',
        periods = [1, 2]
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )
subscribers_daily_df

# percentage change
subscribers_daily_df \
    .augment_pct_change(
        date_column = 'optin_time',
        value_column = 'value',
        periods = 1
    )

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_pct_change(
        date_column = 'optin_time',
        value_column = 'value',
        periods = [1, 2]
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )


# * Fourier Transform ------------------------------------------------------

# - Useful for incorporating seasonality & autocorrelation
# - BENEFIT: Don't need a lag, just need a frequency (based on your time index)

subscribers_daily_df \
    .augment_fourier(
        date_column = 'optin_time',
        periods = 1,  
        max_order = 2
    )

subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_fourier(
        date_column = 'optin_time',
        periods = [1, 2],  
        max_order = 2
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )

# Wavelet transform
# https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
tk.augment_wavelet(
    subscribers_daily_df,
    date_column = 'optin_time',
    value_column = 'value',
    scales = [7],  
    sample_rate = 7,
    method = 'bump'
)
    
subscribers_daily_df \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .augment_wavelet(
        date_column = 'optin_time',
        value_column = 'value',
        scales = [7],  
        sample_rate = 7,
        method = 'bump'
    ) \
    .melt(id_vars = 'optin_time', value_name = "value_ts") \
    .plot_timeseries(
        date_column = 'optin_time',
        value_column = 'value_ts',
        color_column = 'variable',
        smooth = False
    )

# effects on time series regression


# * Confined Interval -----------------------------------------------------

# - Transformation used to confine forecasts to a max/min interval



# Visualization -----------------------------------------------------------

# * Time Series Plot ------------------------------------------------------

# Log Transforms

# * Autocorrelation Function (ACF) Plot -----------------------------------

# * Cross-Correlation Function (CCF) Plot ---------------------------------

# * Smoothing Plot --------------------------------------------------------

# * Boxplots --------------------------------------------------------------

# * Seasonality Plot ------------------------------------------------------

# * Decomposition Plot ----------------------------------------------------

# * Anomaly Detection Plot ------------------------------------------------

# * Time Series Regression Plot -------------------------------------------



















