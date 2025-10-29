
import re
import pandas as pd
import numpy as np
import pandas_flavor as pf
import random
from statsmodels.gam.api import BSplines
from statsforecast import StatsForecast
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse
from utilsforecast.plotting import plot_series


# function to perform data standardization (mean 0, stdev 1)
def standardize(x):
    print(f'mean x = {np.mean(x)}, stdev x = {np.std(x)}')
    return (x - np.mean(x)) / np.std(x)

# function to invert standardization
def inv_standardize(x, mean, stdev):
    return (x * stdev) + mean

# function to perform data log-interval transformation
# log(((x + offset) - a)/(b - (x + offset)))
# a = lower bound, b = upper bound
def log_interval(x, lb = 0, ub = 'auto', offset = 1):
    if (ub == 'auto'):
        ub = np.max(x) * 1.10 
    print(f'lower bound = {lb}, upper bound = {ub}, offset = {offset}')
    return np.log(((x + offset) - lb)/(ub - (x + offset)))

# function to invert log-interval transformation
# (b-a)*(exp(x)) / (1 + exp(x)) + a - offset
def inv_log_interval(x, lb, ub, offset = 1):
    return (ub - lb) * (np.exp(x)) / (1 + np.exp(x)) + lb - offset

# function to add basis splines to the dataframe
@pf.register_dataframe_method
def augment_bsplines(data, column_name = 'ds_index_num', df = 5, degree = 3):
    
    bs = BSplines(
        data[column_name], 
        df = df, 
        degree = degree, 
        include_intercept = False
    )
    bs_df = pd.DataFrame(bs.basis)
    
    col_names = []
    for i in range(1, (len(bs_df.columns) + 1)):
        col_names.append(f'bspline_{i}_degree_{degree}')
    bs_df.columns = col_names
    bs_df.index = data.index

    data_splines = pd.concat([data, bs_df], axis = 1)

    return data_splines

# function to perform time series regression and plotting
@pf.register_dataframe_method
def plot_time_series_regression(data):
    
    # fit linear regression
    fcst = MLForecast(models = LinearRegression(), freq = 'D')
    fcst.fit(data, static_features = [], fitted = True)

    # extract fitted values
    data_fitted_values = fcst.forecast_fitted_values()

    # plot actual vs fitted
    p = data_fitted_values \
        .rename(columns = {'y': 'actual', 'LinearRegression': 'fitted'}) \
        .melt(id_vars = ['unique_id', 'ds']) \
        .plot_timeseries(
            date_column = 'ds', 
            value_column = 'value',
            color_column = 'variable',
            smooth = False
        )

    return p

# function to plot the cross-validation plan
@pf.register_dataframe_method
def plot_cross_validation_plan(
    data, freq, h, 
    n_windows = 1, step_size = 1,
    engine = 'matplotlib'
):

    data = data[['unique_id', 'ds', 'y']]
    sf = StatsForecast(models = [], freq = freq, n_jobs = -1)
    cv_df = sf.cross_validation(
        df = data, h = h, n_windows = n_windows, step_size = step_size
    )

    cv_df.rename(columns = {'y': 'cv_set'}, inplace = True)
    cutoff = cv_df['cutoff'].unique()

    for k in range(len(cutoff)): 
        cv = cv_df[cv_df['cutoff'] == cutoff[k]]
        StatsForecast.plot(
            data, cv.drop('cutoff', axis = 1), 
            engine = engine
        ).show()

# function to perform evaluation on test set
def calibrate_evaluate_plot(
    object, data, h, 
    prediction_intervals = None, level = None,
    loss = None,
    engine = 'matplotlib',
    max_insample_length = None
):

    object_class = str(object.__class__)
    if object_class == "<class 'statsforecast.core.StatsForecast'>":
        cv_res = object.cross_validation(
            df = data, h = h, n_windows = 1,
            prediction_intervals = prediction_intervals, 
            level = level
        )
    elif object_class == "<class 'mlforecast.forecast.MLForecast'>":
        cv_res = object.cross_validation(
            df = data, h = h, n_windows = 1,
            prediction_intervals = prediction_intervals, 
            level = level,
            static_features = []
        )
    elif object_class == "<class 'neuralforecast.core.NeuralForecast'>":
        cv_res = object.cross_validation(
            df = data, n_windows = 1
        )
    else:
        raise Exception(f"Unknown object of class {object_class}")
    
    if loss == 'DistributionLoss':
        cv_res = cv_res \
            .loc[:, ~ cv_res.columns.str.endswith('-median')]
    elif loss == 'MQLoss':
        # remove -median when using MQLoss in DL
        cv_res = cv_res \
            .rename(columns = lambda x: re.sub('-median', '', x))
    else:
        cv_res = cv_res
    
    acc_res = evaluate(
        df = cv_res.drop(columns = 'cutoff'),
        train_df = data,
        metrics = [bias, mae, mape, mse, rmse],
        agg_fn = 'mean'
    )

    if level == None:
        p_res = plot_series(
            df = data.head(n = -h),
            forecasts_df = cv_res.drop('cutoff', axis = 1),
            max_insample_length = max_insample_length,
            engine = engine
        )
    else:
        p_res = plot_series(
            df = data.head(n = -h),
            forecasts_df = cv_res.drop('cutoff', axis = 1),
            level = level,  
            max_insample_length = max_insample_length,
            engine = engine
        )

    res = {'cv_results': cv_res, 'accuracy_table': acc_res, 'plot': p_res}

    return res

# function to get model names from data
@pf.register_dataframe_method
def get_models_name(data):
    r = re.compile(r'(unique_id)|(ds)|(y)|(-lo-)|(-hi-)')
    models_name = [i for i in data.columns if not r.search(i)]
    return models_name

# function to select columns of a dataframe based on regex
@pf.register_dataframe_method
def select_columns(data, regex = None):
    if (regex == None):
        cols_name = ['unique_id', 'ds', 'y']
    else:
        regex = '(^unique_id$)|(^ds$)|(^y$)|' + regex
        r = re.compile(regex)
        cols_name = [i for i in data.columns if r.search(i)]
    return data[cols_name]

# function to transform a dataframe to intermittent (in Nixtla's format)
@pf.register_dataframe_method
def to_intermittent(data, prop_of_zeros = 0.90):

    n = len(data)
    n_with_zeros = int(n * prop_of_zeros)
    ids_with_zero = random.sample(range(1, n), n_with_zeros)
    ids_with_zero.sort()
    data_inter = data.copy()
    data_inter.loc[ids_with_zero, 'y'] = 0

    return data_inter

# function to print accuracy table
@pf.register_dataframe_method
def print_accuracy_table(data, type = 'min'):
    if type == 'min':
        data_res = data \
            .set_index('metric') \
            .style.highlight_min(color = 'green', axis = 1)
    else:
        data_res = data \
            .set_index('metric') \
            .style.highlight_max(color = 'red', axis = 1)
    return data_res

# function to select the best model from accuracy table
@pf.register_dataframe_method
def get_best_model_name(accuracy_data, metric = 'rmse'):
    data_filtered = accuracy_data \
        .melt(id_vars = 'metric') \
        .query("metric == @metric") \
        .reset_index() \
        .drop('index', axis = 1)
    id_min = data_filtered['value'].idxmin()
    model_name = data_filtered.loc[id_min, 'variable']
    return model_name

# function to get the best model forecast results
@pf.register_dataframe_method
def get_best_model_forecast(forecasts_data, accuracy_data, metric = 'rmse'):
    best_name = get_best_model_name(accuracy_data, metric = metric)
    best_forecasts = select_columns(forecasts_data, regex = f'{best_name}')
    return best_forecasts

# function to back transform results
@pf.register_dataframe_method
def back_transform_data(data, params, forecasts_data = None):

    data_back_df = data \
        .transform_columns(
            columns = 'y', 
            transform_func = lambda x: inv_standardize(
                x, params['mean_x'], params['stdev_x']
            )
        ) \
        .transform_columns(
            columns = 'y', 
            transform_func = lambda x: inv_log_interval(
                x, params['lower_bound'], params['upper_bound'], params['offset']
            )
        )

    if forecasts_data is not None:
        cols_to_transform = forecasts_data \
            .drop(['unique_id', 'ds'], axis = 1) \
            .columns
        fcst_back_df = forecasts_data \
            .transform_columns(
                columns = cols_to_transform, 
                transform_func = lambda x: inv_standardize(
                    x, params['mean_x'], params['stdev_x']
                )
            ) \
            .transform_columns(
                columns = cols_to_transform, 
                transform_func = lambda x: inv_log_interval(
                    x, params['lower_bound'], params['upper_bound'], params['offset']
                )
            )
        res = {'data_back': data_back_df, 'forecasts_back': fcst_back_df}
    else:
        res = data_back_df

    return res
