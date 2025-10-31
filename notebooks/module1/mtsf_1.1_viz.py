# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 1.1: Manipulation, Transformation & Visualization -----------------

# Goals:
# - Learn pytimetk data wrangling functionality
# - Commonly used time series transformations
# - Commonly used time series visualizations


# Packages ----------------------------------------------------------------

import sys
sys.path.insert(0, 'src/Python/utils')
from utils import load_data, plot_acf_pacf, plot_seasonal_decompose

import numpy as np
import pandas as pd
import polars as pl
import pytimetk as tk
from scipy.stats import boxcox 
from statsmodels.nonparametric.smoothers_lowess import lowess




# Data --------------------------------------------------------------------

email_df = load_data('data/email/', 'email_prep', ext = '.parquet')
email_df.glimpse()



# Manipulation ------------------------------------------------------------

# * Summarize by Time -----------------------------------------------------

# - Apply commonly used aggregations
# - High-to-Low Frequency

# to weekly
email_df \
    .group_by('unique_id') \
    .tk.summarize_by_time(
        date_column = 'ds', value_column = 'y', 
        freq = 'W', agg_func = 'sum'
    )

# to monthly
email_df \
    .group_by('unique_id') \
    .tk.summarize_by_time(
        date_column = 'ds', value_column = 'y', 
        freq = 'ME', agg_func = 'sum'
    )


# * Pad by Time -----------------------------------------------------------

# - Filling in time series gaps
# - Low-to-High Frequency (un-aggregating)

# fill daily gaps
email_df \
    .group_by('unique_id') \
    .tk.pad_by_time(
        date_column = 'ds',
        freq = 'D',
        start_date = '2018-06-01'
    ) \
    .fill_null(0)

# weekly to daily
email_df \
    .group_by('unique_id') \
    .tk.pad_by_time(
        date_column = 'ds',
        freq = 'D',
        start_date = '2018-06-01'
    ) \
    .fill_null(0) \
    .tk.apply_by_time(
        date_column = 'ds',
        freq = 'W',
        y = lambda df: df['y'].sum() / 7
    )
    

# * Filter by Time --------------------------------------------------------

# - Pare data down before modeling

email_df \
    .tk.filter_by_time(date_column = 'ds', start_date = '2018-11-20')

email_df \
    .tk.filter_by_time(
        date_column = 'ds', 
        start_date = '2018-12-01',
        end_date = '2018-12-31'
    )


# * Apply by Time ---------------------------------------------------------

# - Get change from beginning/end of period

# first, last, mean, median by period
email_df \
    .tk.apply_by_time(
        date_column = 'ds',
        freq = '2W',
        y_mean = lambda df: df['y'].mean(),
        y_median = lambda df: df['y'].median(),
        y_max = lambda df: df['y'].max(),
        y_min = lambda df: df['y'].min()
    )


# * Future Frame ----------------------------------------------------------

# - Forecasting helper

email_df \
    .tk.future_frame(date_column = 'ds', length_out = 10)

future_df = email_df \
    .tk.future_frame(
        date_column = 'ds', 
        length_out = 10, 
        bind_data = False
    )
future_df



# Transformation ----------------------------------------------------------

# * Variance Reduction ----------------------------------------------------

# Log
email_df.with_columns(pl.col('y').log().alias('y'))

email_df \
    .to_pandas() \
    .transform_columns(columns = 'y', transform_func = np.log) # from pytimetk, only pandas backend

# Log + 1
email_df \
    .with_columns(pl.col('y').log1p().alias('y')) \
    .tk.plot_timeseries('ds', 'y', smooth = False)

# - inversion with np.exp() and np.expm1()

# Box-Cox
def boxcox_vec(x, type = 'series'):
    if type == 'series':
        return pl.Series(boxcox(x + 1)[0])
    elif type == 'lambda':
        return boxcox(x + 1)[1] 
    else :
        raise ValueError("type must be 'series' or 'lambda'")

email_df.with_columns(boxcox_vec(email_df['y']).alias('y'))
boxcox_vec(email_df['y'], type = 'lambda')

# - inversion with the lambda y

email_log_df = email_df \
    .select('ds', 'y') \
    .with_columns(pl.col('y').log1p().alias('y')) 


# * Range Reduction -------------------------------------------------------

# - Used in visualization to overlay series
# - Used in ML for models that are affected by feature magnitude (e.g. linear regression)

# Normalization Range (0,1)
def min_max(col = 'y'):
    return ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()))

email_df \
    .with_columns(min_max().alias('y')) \
    .tk.plot_timeseries('ds', 'y', smooth = False)

# Standardization
def standardize(col = 'y'):
    return (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()

email_df \
    .with_columns(standardize().alias('y')) \
    .tk.plot_timeseries('ds', 'y', smooth = False)


# * Smoothing -------------------------------------------------------------

# - Identify trends and cycles
# - Clean seasonality
lowess(
    email_df['y'], 
    range(len(email_df['y'])),
    frac = 0.1
)

def lowess_smoother(x, frac = 0.1):
    return pl.Series(lowess(x, range(len(x)), frac)[:, 1])
    
email_df \
    .with_columns(lowess_smoother(email_df['y']).alias('y')) \
    .tk.plot_timeseries('ds', 'y', smooth = False)


# * Rolling Averages ------------------------------------------------------

# - Common time series operations to visualize trend
# - A simple transformation that can help create improve features
# - Can help with outlier-effect reduction & trend detection
# - Note: Businesses often use a rolling average as a forecasting technique
# A rolling average forecast is usually sub-optimal (good opportunity for you!).

email_df \
    .tk.augment_rolling(
        date_column = 'ds', value_column = 'y',
        window_func = 'mean', window = 7
    )

email_df \
    .tk.augment_rolling(
        date_column = 'ds', value_column = 'y',
        window_func = ['mean', 'std', 'sum'], window = [7, 14, 30]
    )

email_log_df \
    .tk.augment_rolling(
        date_column = 'ds', value_column = 'y',
        window_func = ['mean'], window = [7, 14, 30, 90, 365]
    ) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)

# Exponential Weighted Rolling Average
email_df \
    .tk.augment_ewm(
        date_column = 'ds', value_column = 'y',
        window_func = 'mean', alpha = 0.1
    )

email_log_df \
    .tk.augment_ewm(
        date_column = 'ds', value_column = 'y',
        window_func = 'mean', alpha = 0.1
    ) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries(
        date_column = 'ds', value_column = 'value',
        color_column = 'variable', smooth = False
    )


# * Missing Values Imputation ---------------------------------------------

# - Imputation helps with filling gaps (if needed)

# pd.DataFrame.fillna # fill with a value
# pd.DataFrame.ffill # forward fill
# pd.DataFrame.bfill # backward fill
# pd.DataFrame.interpolate # interpolation

email_miss_df = email_df \
    .select(['unique_id', 'ds', 'y']) \
    .with_columns(
        pl.when(pl.col('y') == 0).then(None).otherwise(pl.col('y')).alias('y')
    ) 
email_miss_df

email_miss_df \
    .with_columns([
        pl.col('y').forward_fill().over('unique_id').alias('y_ffill'),
        pl.col('y').backward_fill().over('unique_id').alias('y_bfill'),
        pl.col('y').interpolate().over('unique_id').alias('y_linear'),
    ])


# * Anomaly Cleaning ------------------------------------------------------

# - Outlier removal helps linear regression detect trend and reduces high leverage points
# WARNING: Make sure you check outliers against events
# - usually there is a reason for large values

# Anomaly detection

email_anom_df = email_log_df \
    .tk.anomalize(
        date_column = 'ds', value_column = 'y', method = 'stl', # 'twitter'
        iqr_alpha = 0.05, max_anomalies = 0.2, 
        clean_alpha = 0.75, clean = 'min-max'
    )
email_anom_df

email_anom_df.tk.plot_anomalies('ds')
email_anom_df.tk.plot_anomalies_decomp('ds')
email_anom_df.tk.plot_anomalies_cleaned('ds')


# * Lags & Differencing ---------------------------------------------------

# - Lags: Often used for feature engineering
# - Lags: Autocorrelation
# - MOST IMPORTANT: Can possibly use lagged variables in a model, if lags are correlated
# - Difference: Used to go from growth to change
# - Difference: Makes a series 'stationary' (potentially)

# lags
email_df \
    .tk.augment_lags(date_column = 'ds', value_column = 'y', lags = 1)

email_log_df \
    .tk.augment_lags(
        date_column = 'ds', value_column = 'y',
        lags = [1, 7, 14, 30, 90]
    ) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)

# differencing
email_df \
    .tk.augment_diffs(date_column = 'ds', value_column = 'y', periods = 1)

email_log_df \
    .tk.augment_diffs(
        date_column = 'ds', value_column = 'y',
        periods = [1, 2]
    ) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)

# percentage change
email_df \
    .tk.augment_pct_change(date_column = 'ds', value_column = 'y', periods = 1)

email_df \
    .select('ds', 'y') \
    .tk.augment_pct_change(
        date_column = 'ds', value_column = 'y',
        periods = [1, 2]
    ) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)


# * Fourier Transform ------------------------------------------------------

# - Useful for incorporating seasonality & autocorrelation
# - BENEFIT: Don't need a lag, just need a frequency (based on your time index)

email_df \
    .tk.augment_fourier(date_column = 'ds', periods = 1, max_order = 2)

email_log_df \
    .tk.augment_fourier(date_column = 'ds', periods = [1, 2], max_order = 2) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)

# Wavelet transform
# https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
email_df \
    .tk.augment_wavelet(
    date_column = 'ds', value_column = 'y',
    scales = [7], sample_rate = 7, method = 'bump'
)
    
email_log_df \
    .tk.augment_wavelet(
        date_column = 'ds', value_column = 'y',
        scales = [1, 7, 14, 30], sample_rate = 7, method = 'bump'
    ) \
    .melt(id_vars = 'ds', value_name = 'value') \
    .tk.plot_timeseries('ds', 'value', color_column = 'variable', smooth = False)


# * Confined Interval -----------------------------------------------------

# - Transformation used to confine forecasts to a max/min interval

def log_interval(col = 'y', lb = 0, ub = 'auto', offset = 1):
    if (ub == 'auto'):
        ub = pl.col(col).max() * 1.10 
    return ((pl.col(col) + offset - lb) / (ub - (pl.col(col) + offset))).log()

email_df \
    .with_columns(log_interval(lb = 0).alias('y')) \
    .tk.plot_timeseries('ds', 'y', smooth = False)

# function to invert log-interval transformation
# (b-a)*(exp(x)) / (1 + exp(x)) + a - offset



# Visualization -----------------------------------------------------------

# * Time Series Plot ------------------------------------------------------

email_df.tk.plot_timeseries('ds', 'y', smooth = False)
email_log_df.tk.plot_timeseries('ds', 'y', smooth = False)


# * Autocorrelation Function (ACF) Plot -----------------------------------

plot_acf_pacf(email_log_df, 'y', lags = 10)

plot_acf_pacf(email_log_df, 'y', lags = 120)


# * Cross-Correlation Function (CCF) Plot ---------------------------------


# * Smoothing Plot --------------------------------------------------------

email_log_df.tk.plot_timeseries('ds', 'y', smooth = True, smooth_frac = 0.2)


# * Boxplots --------------------------------------------------------------


# * Seasonality Plot ------------------------------------------------------


# * Decomposition Plot ----------------------------------------------------

plot_seasonal_decompose(email_log_df, 'y', model='add', period=7)


# * Anomaly Detection Plot ------------------------------------------------

email_anom_df = email_log_df \
    .tk.anomalize(
        date_column = 'ds', value_column = 'y', method = 'stl', # 'twitter'
        iqr_alpha = 0.05, max_anomalies = 0.2, 
        clean_alpha = 0.75, clean = 'min-max'
    )
email_anom_df

email_anom_df.tk.plot_anomalies('ds')
email_anom_df.tk.plot_anomalies_decomp('ds')
email_anom_df.tk.plot_anomalies_cleaned('ds')


# * Time Series Regression Plot -------------------------------------------


