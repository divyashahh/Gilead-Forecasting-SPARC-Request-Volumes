#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[19]:


actuals = pd.read_csv('starburstlongitudinal2.csv')


# In[20]:


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


# In[21]:


df = actuals.copy()
df = df.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df = df[(df.organization_flag == 'Gilead')]
df.head()


# In[22]:


df_north = df[df['gilead_location_level1'] == 'North America (Geographic)']
df_north.head()


# In[23]:


unique_countries = df_north['gilead_location_level2'].unique()
print(unique_countries)


# In[24]:


preserve_columns = ['ds', 'y']
df_north = df_north.filter(items = preserve_columns)
df_north.dtypes
df_north['ds'] = pd.DatetimeIndex(df_north['ds'])
df_north.dtypes
df_north = df_north.set_index(pd.DatetimeIndex(df_north['ds'])).drop('ds', axis=1)
df_north.plot()


# In[25]:


df_north_weekdays = df_north.copy()
df_north_weekdays['day_name'] = df_north.index.day_name()
print(df_north_weekdays.iloc[100:125])


# In[26]:


df_north.resample('W').sum().plot()


# In[27]:


df_north.resample('M').sum().plot()


# In[28]:


df_north_roll = df_north.copy()
df_north_roll['30_day_ma'] = df_north_roll.y.rolling(30, center=True).mean()
df_north_roll['30_std'] = df_north_roll.y.rolling(30, center=True).std()
df_north_roll.plot()


# In[29]:


df_north['day'] = df_north.index.dayofweek
df_north['week'] = df_north.index.strftime('%U').astype(int)
seasonal_plot(df_north, y='y', period='week', freq='day')


# In[30]:


df_north['dayofyear'] = df_north.index.dayofyear
df_north['year'] = df_north.index.year
seasonal_plot(df_north, y='y', period='year', freq='dayofyear')


# In[31]:


plot_periodogram(df_north.y)


# In[32]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq='A', order=10)
df_north = actuals.copy()
df_north = df_north.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df_north = df_north[(df_north.organization_flag == 'Gilead')]
preserve_columns = ['ds', 'y']
df_north = df_north.filter(items = preserve_columns)
df_north['ds'] = pd.DatetimeIndex(df_north['ds'])
df_north.dtypes
df_north = df_north.set_index('ds').to_period('D')

dp = DeterministicProcess(
    index=df_north.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X=dp.in_sample()
X.head()


# In[33]:


y = df_north['y']
model = LinearRegression(fit_intercept=False)
_ = model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=365)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax=y.plot(color='0.25', style='.', title='Seasonal Forecast')
ax=y_pred.plot(ax=ax, label='Seasonal')
ax=y_fore.plot(ax=ax, label='Forecast')
_ = ax.legend


# In[ ]:


prep_df = actuals.copy()
prep_df = prep_df.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
prep_df = prep_df[(prep_df['gilead_location_level1'] == 'North America (Geographic)') & (prep_df['organization_flag'] == 'Gilead')]
preserve_columns = ['ds', 'y']
prep_df = prep_df.filter(items = preserve_columns)
prep_df['ds'] = pd.DatetimeIndex(prep_df['ds'])
prep_df.reset_index(drop=True)
df = prep_df.copy()
 
m = Prophet(
    daily_seasonality = False,
    weekly_seasonality = True,
    yearly_seasonality = True,
    changepoint_prior_scale = 1.0,
    seasonality_prior_scale = 8.0,
    seasonality_mode = 'multiplicative',
    changepoint_range = .8
)
 
m.add_country_holidays(country_name = 'US')
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


# In[35]:


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

