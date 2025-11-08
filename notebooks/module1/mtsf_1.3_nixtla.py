# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture 1.3: Nixtla -----------------------------------------------------

# Goals:
# - Learn the Nixtla Workflow
# - Understand Accuracy Measurements
# - Understand the Forecast Horizon & Confidence Intervals
# - Understand Refitting

# Challenges:
# - Challenge (Optional) - Nixtla



# Packages ----------------------------------------------------------------

import pickle
import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    plot_cross_validation_plan, select_columns,
    back_transform_data, back_transform_forecasts
)
import pytimetk as tk
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse
from utilsforecast.plotting import plot_series



# Data & Artifacts --------------------------------------------------------

with open('data/email/artifacts/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']


# * Recipes ---------------------------------------------------------------

y_df = select_columns(data_prep_df)
forecast_y_df = select_columns(forecast_df).drop('y')

y_df.tk.plot_timeseries('ds', 'y', smooth = False)


# * Forecast Horizon ------------------------------------------------------

horizon = 7 * 8 # 8 weeks


# * Cross-validation Plan -------------------------------------------------

# With Nixtla's workflow there is no need to split data before 
# since validation is performed directly through the cross_validation method
# however it is always useful to visualize the validation plan 
# for that one can just cross-validate a naive model to obtain the 
# validation plan (cutoffs dates)
plot_cross_validation_plan(y_df, freq = '1d', h = horizon, n_windows = 1, step_size = 1)



# Nixtla ---------------------------------------------------------------

# https://nixtlaverse.nixtla.io/
# https://github.com/Nixtla


# * Engines (Algorithms' Specification) -----------------------------------

from statsforecast.models import (AutoARIMA, HoltWinters)

# create the model list with models' parameters
models = [
    AutoARIMA(season_length = 7),
    HoltWinters(season_length = 7, error_type = 'A')
]

# initialize the Nixtla forecast object
# this may be one of StatsForecast, MLForecast or NeuralForecast
sf = StatsForecast(models = models, freq = '1d', n_jobs = -1)


# * Evaluation via Cross Validation ---------------------------------------

# cross-validate the models with cross_validation
cv_res = sf.cross_validation(df = y_df, h = horizon, n_windows = 1)
cv_res.head()

acc_res = evaluate(
    df = cv_res.drop('cutoff'),
    train_df = y_df,
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)    
acc_res 

plot_series(
    df = y_df.head(n = -horizon), forecasts_df = cv_res.drop('cutoff'),
    max_insample_length = horizon * 2, engine = 'plotly'
)


# * Refitting ---------------------------------------------------------------

# on the whole dataset 
sf.fit(df = y_df)


# * Predicting ------------------------------------------------------------

# out-of-sample point forecasting
preds_df = sf.predict(h = horizon)
plot_series(
    df = y_df, forecasts_df = preds_df,
    max_insample_length = horizon * 2, engine = 'plotly'
)

# out-of-sample probabilistic forecasting
preds_df_prob = sf.predict(h = horizon, level = [80, 95])
plot_series(
    df = y_df, forecasts_df = preds_df_prob, level = [80, 95],
    max_insample_length = horizon * 2, engine = 'plotly'
)


# Extra Topics ------------------------------------------------------------

# * Residuals' Diagnostics ------------------------------------------------

# - Explore in-sample and out-of-sample residuals

# Auto-ARIMA results
result = sf.fitted_[0,0].model_
print(result.keys())
print(result['arma'])

resids = pd.DataFrame(result.get('residuals'), columns = ['resids'])
resids.head()


fig, axs = plt.subplots(nrows = 2, ncols = 2)
# plot[1,1]
resids['resids'].plot(ax = axs[0, 0])
axs[0, 0].set_title('Residuals');
# plot[1,2]
sns.distplot(resids['resids'], ax = axs[0, 1]);
axs[0, 1].set_title('Density plot - Residual');

# plot[2,1]
stats.probplot(resids['resids'], dist = 'norm', plot = axs[1, 0])
axs[1, 0].set_title('Plot Q-Q')

# plot[2,2]
plot_acf(resids['resids'], lags = 24, ax = axs[1, 1])
axs[1, 1].set_title('Autocorrelation');

plt.show();


# * Expedited Forecasting -------------------------------------------------

# - Fitted on Full dataset (No Train/Test)
# - Forecast directly without confidence intervals
fcst_sf = sf.forecast(df = y_df, h = horizon)
fcst_sf.head()
plot_series(
    df = y_df, forecasts_df = fcst_sf,
    max_insample_length = horizon * 2, engine = 'plotly'
)


# * Back Transformation ---------------------------------------------------

back_y_df = back_transform_data(df = y_df, params = params, col = 'y')
back_preds_df = back_transform_forecasts(df = preds_df, params = params)
plot_series(
    df = back_y_df, forecasts_df = back_preds_df,
    max_insample_length = horizon * 2, engine = 'plotly'
)


# * Conformal Intervals ---------------------------------------------------

levels = [80, 95]
intervals = ConformalIntervals(h = horizon, n_windows = 3)
# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.

# engines
conformal_models = [
    AutoARIMA(season_length = 7, prediction_intervals = intervals),
    HoltWinters(season_length = 7, error_type = 'A', prediction_intervals = intervals)
]
conformal_sf = StatsForecast(models = conformal_models, freq = '1d', n_jobs = -1)

# fit
conformal_fit_sf = conformal_sf.fit(df = y_df)

# out-of-sample conformal probabilistic forecasting
conformal_preds_df_prob = conformal_fit_sf.predict(h = horizon, level = [80, 95])
plot_series(
    df = y_df, forecasts_df = conformal_preds_df_prob, level = [80, 95],
    max_insample_length = horizon * 2, engine = 'plotly'
)