# Modern Time Series Forecasting with Python ----
# Marco Zanotti

# Lecture E.1: Intermittency ----------------------------------------------

# Goals:
# - Baseline Models
# - SARIMAX
# - ETS
# - Theta
# - TBATS
# - MSTL
# - PROPHET
# - BONUS: Intermittent Demand Models



# Packages ----------------------------------------------------------------

import pickle
import sys
sys.path.insert(0, 'src/Python/utils')
from utils import (
    plot_cross_validation_plan, select_columns, calibrate_evaluate_plot,
    print_accuracy_table, back_transform_data, to_intermittent
)
import random
import pytimetk as tk

from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals



# Data & Artifacts --------------------------------------------------------

with open('data/email/artifacts/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
params = data_loaded['transform_params']

# classical time series models just need the target variable and date
# and back transform to original series
y_df = back_transform_data(select_columns(data_prep_df), params)
forecast_y_df = select_columns(forecast_df).drop('y')

# create intermittent series
random.seed(1992)
inter_df = to_intermittent(y_df, prop_of_zeros = 0.90)

inter_df.tk.plot_timeseries('ds', 'y', smooth = False)


# * Forecast Horizon ------------------------------------------------------

horizon = 7 * 8 # 8 weeks


# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
# Conformal intervals
intervals = ConformalIntervals(h = horizon, n_windows = 2)
# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.


# * Cross-validation Plan -------------------------------------------------

# with Nixtla's workflow there is no need to split data before 
# since validation is performed directly through the cross_validation method
# however it is always useful to visualize the validation plan 
# for that one can just cross-validate a naive model to obtain the 
# validation plan (cutoffs dates)
plot_cross_validation_plan(
    inter_df, freq = '1d', h = horizon, 
    n_windows = 1, step_size = 1
)

plot_cross_validation_plan(
    inter_df, freq = '1d', h = horizon, 
    n_windows = 6, step_size = 14
)



# Sparse & Intermittent Models --------------------------------------------

from statsforecast.models import (
    CrostonClassic as Croston, 
    CrostonOptimized,
    CrostonSBA,
    TSB,
    ADIDA, 
    IMAPA
)

# * Engines ---------------------------------------------------------------

models_inter = [
    Croston(),
    CrostonOptimized(),
    CrostonSBA(),
    TSB(alpha_d = 0.5, alpha_p = 0.5),
    ADIDA(),
    IMAPA()
]
sf_inter = StatsForecast(
    models = models_inter,
    freq = '1d', 
    n_jobs = -1,
)

# * Evaluation ------------------------------------------------------------

cv_res_inter = calibrate_evaluate_plot(
    object = sf_inter, df = inter_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2, plot_level = False 
)
cv_res_inter['cv_results']
print_accuracy_table(cv_res_inter['accuracy_table'])
cv_res_inter['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

# if you need model parameters use .fit and .predict 
# otherwise use .forecast (optimized to run also on clusters)
# P.S. specify fitted = True to store the fitted values

# fit_baseline = sf_baseline.fit(
#     df = data_prep_df, prediction_intervals = intervals
# )
# preds_df_baseline = fit_baseline.predict(
#     h = horizon, level = levels
# )
fcst_df_inter = sf_inter.forecast(
    df = inter_df, h = horizon,
    prediction_intervals = intervals, 
    level = levels
)
fcst_df_inter

sf_inter.plot(
    inter_df, fcst_df_inter,
    max_insample_length = horizon * 2,
    engine = 'plotly'
).show()

for nm in sf_inter.models:
    sf_inter.plot(
        inter_df, 
        fcst_df_inter,
        models = [str(nm)], 
        level = levels,
        max_insample_length = horizon * 2,
        engine = 'plotly'
    ).show()