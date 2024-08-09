#!/usr/bin/env python
# coding: utf-8

# In[1]:


import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import certifi, csv, glob, itertools, logging, os, progressbar, re, ssl, time, trino, yaml
import plotly.offline as poff
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import init_notebook_mode
from colorama import Fore, Back, Style
from matplotlib import cycler
from multiprocessing.pool import ThreadPool
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics 
from prophet.plot import add_changepoints_to_plot, plot, plot_components, plot_cross_validation_metric, plot_forecast_component, plot_plotly, plot_seasonality, plot_weekly, plot_yearly, seasonality_plot_df
from pyspark.sql import SparkSession
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)
logging.getLogger('prophet').setLevel(logging.WARNING)
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
pio.templates.default = 'plotly_dark'
poff.init_notebook_mode()


# In[2]:


actuals = pd.read_csv('starburstlongitudinal2.csv')


# In[3]:


def fetch_rows(cur):
    pool = ThreadPool(processes=1)
    async_result = pool.apply_async(cur.fetchall, ())
    bar = progressbar.ProgressBar(maxval=100, widgets=['Query Running: ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar_start = False
    
    while not async_result.ready():
        if cur.stats['scheduled']:
            if not bar_start:
                bar.start()
                bar_start=True
            perc = round((cur.stats['completedSplits']*100.0)/(cur.stats['totalSplits']),2)
            bar.update(int(perc))
            time.sleep(1)
        else:
            perc = '0'
            print(Fore.RED, cur.stats['state']+'-'+str(perc)+'%')
            time.sleep(4)
    if bar_start:
        bar.finish()
    print(Fore.RED, cur.stats['state']+'-'+str(cur.stats.get('progressPercentage','')))
    
    return_val = async_result.get()
    return return_val

from pathlib import Path
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# In[4]:


df_china = actuals.copy()
df_china = df_china.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df_china.head()


# In[5]:


df_china = df_china[df_china['gilead_location_level2'] == 'China (Geographic)']
df_china.head()


# In[6]:


unique_countries = df_china['gilead_location_level2'].unique()
print(unique_countries)


# In[7]:


preserve_columns = ['ds', 'y']
df_china = df_china.filter(items = preserve_columns)
df_china.dtypes
df_china['ds'] = pd.DatetimeIndex(df_china['ds'])
df_china.dtypes
df_china = df_china.set_index(pd.DatetimeIndex(df_china['ds'])).drop('ds', axis=1)
df_china.plot()


# In[8]:


df_china_weekdays = df_china.copy()
df_china_weekdays['day_name'] = df_china.index.day_name()
print(df_china_weekdays.iloc[100:125])


# In[9]:


df_china.resample('W').sum().plot()


# In[10]:


df_china.resample('M').sum().plot()


# In[11]:


df_china_roll = df_china.copy()
df_china_roll['30_day_ma'] = df_china_roll.y.rolling(30, center=True).mean()
df_china_roll['30_std'] = df_china_roll.y.rolling(30, center=True).std()
df_china_roll.plot()


# In[12]:


df_china['day'] = df_china.index.dayofweek
df_china['week'] = df_china.index.strftime('%U').astype(int)
seasonal_plot(df_china, y='y', period='week', freq='day')


# In[13]:


df_china['dayofyear'] = df_china.index.dayofyear
df_china['year'] = df_china.index.year
seasonal_plot(df_china, y='y', period='year', freq='dayofyear')


# In[14]:


plot_periodogram(df_china.y)


# In[15]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq='A', order=10)
df_china = actuals.copy()
df_china = df_china.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df_china = df_china[(df_china.organization_flag == 'Gilead')]
preserve_columns = ['ds', 'y']
df_china = df_china.filter(items = preserve_columns)
df_china['ds'] = pd.DatetimeIndex(df_china['ds'])
df_china.dtypes
df_china = df_china.set_index('ds').to_period('D')

dp = DeterministicProcess(
    index=df_china.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X=dp.in_sample()
X.head()


# In[16]:


plot_periodogram(df_china.y)


# In[17]:


y = df_china['y']
model = LinearRegression(fit_intercept=False)
_ = model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=365)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax=y.plot(color='0.25', style='.', title='Seasonal Forecast')
ax=y_pred.plot(ax=ax, label='Seasonal')
ax=y_fore.plot(ax=ax, label='Forecast')
_ = ax.legend


# In[18]:


prep_df = actuals.copy()
prep_df = prep_df.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
display(prep_df)
prep_df = prep_df[prep_df['gilead_location_level2'] == 'China (Geographic)']
display(prep_df)
prep_df = prep_df.groupby(['ds'])['y'].sum().reset_index(name ='y')
display(prep_df)
cleaned_actuals = prep_df.copy()
prep_df['ds'] = pd.DatetimeIndex(prep_df['ds'])
prep_df.reset_index(drop=True)
df = prep_df.copy()
 
m = Prophet(
    daily_seasonality = False,
    weekly_seasonality = True,
    yearly_seasonality = True,
    changepoint_prior_scale = 0.065,
    seasonality_prior_scale = 5.0,
    seasonality_mode = 'multiplicative',
    changepoint_range = 0.8
)
 
m.add_country_holidays(country_name = 'CN')
m.fit(df)
cp = df.loc[df['ds'].isin(m.changepoints)]
cp_p = m.params
future = m.make_future_dataframe(periods = 180) #define number of forecasts
forecast = m.predict(future)
 
plt.style.use('ggplot')
fig1 = m.plot(forecast, uncertainty = True, xlabel = 'Date', ylabel = 'Fits/Forecast', figsize=(20, 7))
ax = fig1.gca()
ax.set_ylim([0, None])
ax.xaxis.grid(False)
ax.yaxis.grid(linestyle='--')
a = add_changepoints_to_plot(fig1.gca(), m, forecast)
fig2 = m.plot_components(forecast, figsize=(20, 10))
for ax in fig2.axes:
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_xlabel(None)


# In[19]:


from prophet.diagnostics import cross_validation

df_cv = cross_validation(m, initial='180 days', period='90 days', horizon = '90 days')

from prophet.diagnostics import performance_metrics

cutoffs = pd.to_datetime(['2024-03-01', '2024-01-01', '2023-10-01'])

df_p = performance_metrics(df_cv)

df_p.head()

df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='90 days')

from prophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')

 

import itertools
import numpy as np
import pandas as pd


param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(df)  # Fit model with given params
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)


# In[ ]:


import itertools
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
logging.getLogger('fbprophet').disabled = True
 
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}
 
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mape = []  # Store the RMSEs for each params here
 
# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(df)  # Fit model with given params
    df_cv = cross_validation(m, initial='366 days', period='90 days', horizon = '90 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    mape.append(df_p['mape'].values[0])
 
# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['MAPE'] = mape
display(tuning_results)
best_params = all_params[np.argmin(mape)]
display(best_params)
fig = plot_cross_validation_metric(df_cv, metric='mape')


# In[ ]:





# In[ ]:


import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_rank, plot_timeline
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
def objective_prophet(opt):
    changepoint_prior_scale = opt.suggest_float('changepoint_prior_scale', low=.01, high=1, step=.005)
    seasonality_prior_scale = opt.suggest_float('seasonality_prior_scale', low=.5, high=10, step=.5)
    seasonality_mode = opt.suggest_categorical('seasonality_mode', ['multiplicative', 'additive'])
 
    m = Prophet(daily_seasonality = False,
                          weekly_seasonality = True,
                          yearly_seasonality = True,
                          changepoint_prior_scale = changepoint_prior_scale,
                          seasonality_prior_scale = seasonality_prior_scale,
                          seasonality_mode = seasonality_mode
                         )
   
    m.add_country_holidays(country_name = 'US')
    m.fit(df)
 
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='90 days', parallel='threads')
    df_p = performance_metrics(df_cv, rolling_window=1)
    return df_p['mape'].values[0]
 
from datetime import datetime, timedelta
max_date = pd.to_datetime(max(df['ds'])).date()
cutoff_1 = (max_date + pd.DateOffset(days=-90)).date()
cutoff_2 = (cutoff_1 + pd.DateOffset(days=-30)).date()
cutoff_3 = (cutoff_2 + pd.DateOffset(days=-30)).date()
cutoff_4 = (cutoff_3 + pd.DateOffset(days=-30)).date()
cutoff_5 = (cutoff_4 + pd.DateOffset(days=-30)).date()
cutoff_6 = (cutoff_5 + pd.DateOffset(days=-30)).date()
print(Fore.RED, 'Latest entry: ' + str(max_date))
print(Fore.WHITE, 'Cutoffs: ' + str(cutoff_1) + ', ' + str(cutoff_2) + ', ' + str(cutoff_3) + ', ' + str(cutoff_4) + ', ' + str(cutoff_5) + ', ' + str(cutoff_6))
cutoffs = pd.to_datetime([cutoff_1, cutoff_2, cutoff_3, cutoff_4, cutoff_5, cutoff_6])
# Find the best parameters
trial_count = 500
start = time.time()
now = datetime.now()
print(Fore.CYAN, f'Beginning Bayesian optimization of RMSE for dataframe across {trial_count} trials at: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
study_prophet = optuna.create_study(sampler=TPESampler(), direction='minimize')
study_prophet.optimize(objective_prophet, n_trials=trial_count, show_progress_bar=True)
total_time = time.time() - start
#output best parameters,
opt_params = pd.DataFrame([study_prophet.best_params])
display(opt_params)


# In[20]:


display(df_china)


# In[21]:


display(forecast)
columns_to_keep=['ds','yhat']
filtered_forecast=forecast.filter(items=columns_to_keep)
display(filtered_forecast)
filtered_forecast = filtered_forecast.rename(columns={'yhat':'yhat_apac_china'})
display(filtered_forecast)
filtered_forecast.dtypes
filtered_forecast=filtered_forecast[(filtered_forecast['ds'] <= '2024-05-31')]
display(filtered_forecast)


# In[22]:


display(df)
max_date = pd.to_datetime(max(df['ds'])).date()
display(max_date)


# In[33]:


df['ds'] = pd.to_datetime(df['ds'])
merged_actuals = df.merge(filtered_forecast, on='ds', how='left')
display(merged_actuals)
merged_actuals.dtypes
merged_actuals['yhat_apac_china'] = merged_actuals['yhat_apac_china'].clip(lower=0)
display(merged_actuals)


# In[34]:


plot_df = merged_actuals
#plot_df = plot_df.set_index('ds')
plt.plot(plot_df.index, plot_df['y'])
plt.plot(plot_df.index, plot_df['yhat_apac_china'])


# In[35]:


plot_df['accuracy_by_day'] = 1-abs(plot_df['y']/plot_df['yhat_apac_china'])


# In[36]:


display(plot_df)


# In[37]:


plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
plot_df.dropna(subset=['accuracy_by_day'], how='all', inplace=True)
plot_df['accuracy_by_day'].mean()
plot_df.mean(axis=0)


# In[38]:


display(merged_actuals)


# In[39]:


merged_actuals = merged_actuals.rename(columns={'y':'y_apac_china'})
display(merged_actuals)


# In[40]:


merged_actuals.to_csv('apac_china.csv', index=False, header=True)

