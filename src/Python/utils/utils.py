import os
import re
import pandas as pd
import polars as pl
import numpy as np
import random
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsforecast import StatsForecast
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse
from utilsforecast.plotting import plot_series


# function to load data
def load_data(file_path, file_name, ext = '.parquet'):

    full_path = f'{file_path}{file_name}{ext}'

    if ext == '.parquet':
        res_df = pl.read_parquet(full_path)
    elif ext == '.csv':
        res_df = pl.read_csv(full_path)
    else:
        raise(f'Unsupported file extension {ext}. Only .parquet and .csv are allowed')

    return res_df

# function to save data
def save_data(data, file_path, file_name, ext = '.parquet'):
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    full_path = f'{file_path}{file_name}{ext}'

    if ext == '.parquet':
        data.to_parquet(full_path)
    elif ext == '.csv':
        data.to_csv(full_path)
    else:
        raise(f'Unsupported file extension {ext}. Only .parquet and .csv are allowed')

# function to perform data standardization (mean 0, stdev 1)
def standardize(col = 'y', mean = 0, stdev = 1):
    return (pl.col(col) - mean) / stdev

# function to invert standardization
def inv_standardize(col = 'y', mean = 0, stdev = 1):
    return (pl.col(col) * stdev) + mean

# function to perform data log-interval transformation (Pandas)
# log(((x + offset) - a)/(b - (x + offset)))
# a = lower bound, b = upper bound
def log_interval(col = 'y', lb = 0, ub = 'auto', offset = 1):
    if (ub == 'auto'):
        ub = pl.col(col).max() * 1.10 
    return ((pl.col(col) + offset - lb) / (ub - (pl.col(col) + offset))).log()

# function to invert log-interval transformation
# (b-a)*(exp(x)) / (1 + exp(x)) + a - offset
def inv_log_interval(col = 'y', lb = 0, ub = None, offset = 1):
    return (ub - lb) * (pl.col(col).exp()) / (1 + pl.col(col).exp()) + lb - offset

# function to plot ACF and PACF
def plot_acf_pacf(df, column, lags):

    x = None
    try:
        if isinstance(df, pl.DataFrame):
            x = df.get_column(column).to_numpy()
        elif isinstance(df, pl.Series):
            x = df.to_numpy()
    except Exception:
        pass

    if x is None:
        # pandas or array-like
        if hasattr(df, "columns") and column in getattr(df, "columns", []):
            x = df[column].to_numpy()
        else:
            x = np.asarray(df)

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    plot_acf(x, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF: {column}")
    plot_pacf(x, lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF: {column}")
    fig.tight_layout()
    return fig

# function to perform seasonal decomposition and plotting
def plot_seasonal_decompose(df, column, model = 'add', period = None):
    
    pdf = df.select(['ds', column]).to_pandas()
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    s = pdf.sort_values('ds').set_index('ds')[column]

    res = seasonal_decompose(s, model=model, period=period)
    fig = res.plot()
    return fig

# function to perform time series regression and plotting
def plot_time_series_regression(df):
    
    # fit linear regression
    fcst = MLForecast(models = LinearRegression(), freq = '1d')
    fcst.fit(df, static_features = [], fitted = True)

    # extract fitted values
    data_fitted_values = fcst.forecast_fitted_values()

    # plot actual vs fitted
    p = data_fitted_values \
        .rename({'y': 'actual', 'LinearRegression': 'fitted'}) \
        .melt(id_vars = ['unique_id', 'ds']) \
        .tk.plot_timeseries(
            date_column = 'ds', value_column = 'value',
            color_column = 'variable', smooth = False
        )

    return p

# function to plot the cross-validation plan
def plot_cross_validation_plan(
    df, freq, h, 
    n_windows = 1, step_size = 1,
    engine = 'plotly'
):

    df = df.select('unique_id', 'ds', 'y')
    sf = StatsForecast(models = [], freq = freq, n_jobs = -1)
    cv_df = sf.cross_validation(
        df = df, h = h, n_windows = n_windows, step_size = step_size
    )

    cv_df = cv_df.rename({'y': 'cv_set'})
    cutoff = cv_df['cutoff'].unique()

    for k in range(len(cutoff)): 
        cv = cv_df.filter(pl.col('cutoff') == cutoff[k]).drop('cutoff')
        StatsForecast.plot(df, cv, engine = engine).show()

# function to perform evaluation on test set
def calibrate_evaluate_plot(
    object, df, h, 
    prediction_intervals = None, 
    level = None,
    loss = None,
    engine = 'plotly',
    max_insample_length = None, 
    plot_level = False
):

    object_class = str(object.__class__)
    if object_class == "<class 'statsforecast.core.StatsForecast'>":
        cv_res = object.cross_validation(
            df = df, h = h, n_windows = 1,
            prediction_intervals = prediction_intervals, 
            level = level
        )
    elif object_class == "<class 'mlforecast.forecast.MLForecast'>":
        cv_res = object.cross_validation(
            df = df, h = h, n_windows = 1,
            prediction_intervals = prediction_intervals, 
            level = level,
            static_features = []
        )
    elif object_class == "<class 'neuralforecast.core.NeuralForecast'>":
        cv_res = object.cross_validation(
            df = df, n_windows = 1
        )
    else:
        raise Exception(f"Unknown object of class {object_class}")
    
    if isinstance(cv_res, pd.DataFrame):
        
        if loss == 'DistributionLoss':
            cv_res = cv_res \
                .loc[:, ~ cv_res.columns.str.endswith('-median')]
        elif loss == 'MQLoss':
            # remove -median when using MQLoss in DL
            cv_res = cv_res \
                .rename(columns = lambda x: re.sub('-median', '', x))
        
        cv_res_no_cutoff = cv_res.drop('cutoff', axis = 1)
    
    else:

        cv_res_no_cutoff = cv_res.drop('cutoff')
    

    acc_res = evaluate(
        df = cv_res_no_cutoff,
        train_df = df,
        metrics = [bias, mae, mape, mse, rmse],
        agg_fn = 'mean'
    )

    if not plot_level:
        level = None

    if level is None:
        p_res = plot_series(
            df = df.head(n = -h),
            forecasts_df = cv_res_no_cutoff,
            max_insample_length = max_insample_length,
            engine = engine
        )
    else:
        p_res = plot_series(
            df = df.head(n = -h),
            forecasts_df = cv_res_no_cutoff,
            level = level,  
            max_insample_length = max_insample_length,
            engine = engine
        )

    res = {'cv_results': cv_res, 'accuracy_table': acc_res, 'plot': p_res}

    return res

# function to print accuracy table
def print_accuracy_table(df, type = 'min'):
    
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    
    if type == 'min':
        data_res = df \
            .set_index('metric') \
            .style.highlight_min(color = 'green', axis = 1)
    else:
        data_res = df \
            .set_index('metric') \
            .style.highlight_max(color = 'red', axis = 1)   
    return data_res

# function to select columns of a dataframe based on regex
def select_columns(df, regex = None):
    if (regex == None):
        cols_name = ['unique_id', 'ds', 'y']
    else:
        regex = '(^unique_id$)|(^ds$)|(^y$)|' + regex
        r = re.compile(regex)
        cols_name = [i for i in df.columns if r.search(i)]
    return df[cols_name]

# function to select the best model from accuracy table
def get_best_model_name(accuracy_df, metric = 'rmse'):

    if isinstance(accuracy_df, pd.DataFrame):
        accuracy_df = pl.from_pandas(accuracy_df)

    model_name = accuracy_df \
        .melt(id_vars = 'metric') \
        .filter(pl.col('metric') == metric) \
        .filter(pl.col('value') == pl.col('value').min()) \
        .select('variable') \
        .item()
    return model_name

# function to get the best model forecast results
def get_best_model_forecast(forecasts_data, accuracy_data, metric = 'rmse'):
    best_name = get_best_model_name(accuracy_data, metric = metric)
    best_forecasts = select_columns(forecasts_data, regex = f'{best_name}')
    return best_forecasts

# function to back transform results
def back_transform_data(df, params, col = 'y'):

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    back_df = df \
        .with_columns(
            inv_standardize(
                col = col, 
                mean = params['mean_y'], 
                stdev = params['stdev_y']
            ).alias(col)
        ) \
        .with_columns(
            inv_log_interval(
                col = col, 
                lb = params['lower_bound'], 
                ub = params['upper_bound'], 
                offset = params['offset']
            ).alias(col)
        )

    return back_df

# function to back transform forecast results
def back_transform_forecasts(df, params):

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    cols_to_transform = df.drop('unique_id', 'ds').columns
    back_df = df
    for col in cols_to_transform:
        back_df = back_transform_data(back_df, params, col)

    return back_df

# function to get model names from data
def get_models_name(data):
    r = re.compile(r'(unique_id)|(ds)|(y)|(-lo-)|(-hi-)')
    models_name = [i for i in data.columns if not r.search(i)]
    return models_name

# function to transform a dataframe to intermittent (in Nixtla's format)
def to_intermittent(df, prop_of_zeros = 0.90):

    n = len(df)
    n_with_zeros = int(n * prop_of_zeros)
    ids_with_zero = random.sample(range(1, n), n_with_zeros)
    ids_with_zero.sort()
    df = df
    inter_df = df \
        .with_row_count("row_nr") \
        .with_columns(
            pl.when(pl.col("row_nr").is_in(ids_with_zero))
            .then(pl.lit(0))
            .otherwise(pl.col("y"))
            .alias("y")
        ) \
        .drop("row_nr")

    return inter_df






