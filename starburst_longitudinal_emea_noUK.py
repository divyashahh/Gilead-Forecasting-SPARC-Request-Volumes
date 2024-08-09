#!/usr/bin/env python
# coding: utf-8

# In[56]:


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


# In[57]:


actuals = pd.read_csv('starburstlongitudinal2.csv')


# In[58]:


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


# In[59]:


df_no_uk = actuals.copy()
df_no_uk.head()


# In[60]:


df_no_uk = actuals[(actuals['gilead_location_level1'] == 'EMEA (Geographic)') & (actuals['gilead_location_level2'] != 'United Kingdom (Geographic)')]
df_no_uk = df_no_uk.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df_no_uk.head()


# In[61]:


unique_countries = df_no_uk['gilead_location_level2'].unique()
print(unique_countries)


# In[62]:


preserve_columns = ['ds', 'y']
df_no_uk = df_no_uk.filter(items = preserve_columns)
df_no_uk.dtypes
df_no_uk['ds'] = pd.DatetimeIndex(df_no_uk['ds'])
df_no_uk.dtypes
df_no_uk = df_no_uk.set_index(pd.DatetimeIndex(df_no_uk['ds'])).drop('ds', axis=1)
df_no_uk.plot()


# In[63]:


df_no_uk_weekdays = df_no_uk.copy()
df_no_uk_weekdays['day_name'] = df_no_uk.index.day_name()
print(df_no_uk_weekdays.iloc[100:125])


# In[64]:


df_no_uk.resample('W').sum().plot()


# In[65]:


df_no_uk.resample('M').sum().plot()


# In[66]:


df_no_uk_roll = df_no_uk.copy()
df_no_uk_roll['30_day_ma'] = df_no_uk_roll.y.rolling(30, center=True).mean()
df_no_uk_roll['30_std'] = df_no_uk_roll.y.rolling(30, center=True).std()
df_no_uk_roll.plot()


# In[67]:


df_no_uk['day'] = df_no_uk.index.dayofweek
df_no_uk['week'] = df_no_uk.index.strftime('%U').astype(int)
seasonal_plot(df_no_uk, y='y', period='week', freq='day')


# In[68]:


df_no_uk['dayofyear'] = df_no_uk.index.dayofyear
df_no_uk['year'] = df_no_uk.index.year
seasonal_plot(df_no_uk, y='y', period='year', freq='dayofyear')


# In[69]:


plot_periodogram(df_no_uk.y)


# In[70]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq='A', order=10)
df_no_uk = actuals.copy()
df_no_uk = df_no_uk.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df_no_uk = df_no_uk[(df_no_uk.organization_flag == 'Gilead')]
preserve_columns = ['ds', 'y']
df_no_uk = df_no_uk.filter(items = preserve_columns)
df_no_uk['ds'] = pd.DatetimeIndex(df_no_uk['ds'])
df_no_uk.dtypes
df_no_uk = df_no_uk.set_index('ds').to_period('D')

dp = DeterministicProcess(
    index=df_no_uk.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X=dp.in_sample()
X.head()


# In[71]:


plot_periodogram(df_no_uk.y)


# In[72]:


y = df_no_uk['y']
model = LinearRegression(fit_intercept=False)
_ = model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=365)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax=y.plot(color='0.25', style='.', title='Seasonal Forecast')
ax=y_pred.plot(ax=ax, label='Seasonal')
ax=y_fore.plot(ax=ax, label='Forecast')
_ = ax.legend


# In[84]:


prep_df = actuals.copy()
prep_df = prep_df.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
display(prep_df)
prep_df = prep_df[(prep_df.gilead_location_level1 == 'EMEA (Geographic)') & (prep_df.gilead_location_level2 != 'United Kingdom (Geographic)')]
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
    changepoint_prior_scale = 0.95,
    seasonality_prior_scale = 6.5,
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


# In[74]:


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
trial_count = 100
start = time.time()
now = datetime.now()
print(Fore.CYAN, f'Beginning Bayesian optimization of RMSE for dataframe across {trial_count} trials at: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
study_prophet = optuna.create_study(sampler=TPESampler(), direction='minimize')
study_prophet.optimize(objective_prophet, n_trials=trial_count, show_progress_bar=True)
total_time = time.time() - start
#output best parameters,
opt_params = pd.DataFrame([study_prophet.best_params])
display(opt_params)


# In[85]:


display(forecast)
columns_to_keep=['ds','yhat']
filtered_forecast=forecast.filter(items=columns_to_keep)
display(filtered_forecast)
filtered_forecast = filtered_forecast.rename(columns={'yhat':'yhat_emea_no_uk'})
display(filtered_forecast)
filtered_forecast.dtypes
filtered_forecast=filtered_forecast[(filtered_forecast['ds'] <= '2024-05-31')]
display(filtered_forecast)


# In[98]:


display(df)
max_date = pd.to_datetime(max(df['ds'])).date()
display(max_date)


# In[99]:


df['ds'] = pd.to_datetime(df['ds'])
merged_actuals = df.merge(filtered_forecast, on='ds', how='left')
display(merged_actuals)
merged_actuals.dtypes
merged_actuals['yhat_emea_no_uk'] = merged_actuals['yhat_emea_no_uk'].clip(lower=0)
display(merged_actuals)


# In[100]:


plot_df = merged_actuals
#plot_df = plot_df.set_index('ds')
plt.plot(plot_df.index, plot_df['y'])
plt.plot(plot_df.index, plot_df['yhat_emea_no_uk'])


# In[101]:


plot_df['accuracy_by_day'] = 1-abs(plot_df['y']/plot_df['yhat_emea_no_uk'])


# In[102]:


plt.plot(plot_df.index, plot_df['accuracy_by_day'])


# In[103]:


display(plot_df)


# In[104]:


plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
plot_df.dropna(subset=['accuracy_by_day'], how='all', inplace=True)
plot_df['accuracy_by_day'].mean()
plot_df.mean(axis=0)


# In[105]:


plt.plot(plot_df.index, plot_df['accuracy_by_day'])


# In[106]:


display(merged_actuals)


# In[107]:


merged_actuals = merged_actuals.rename(columns={'y':'y_emea_no_uk'})
display(merged_actuals)


# In[108]:


merged_actuals.to_csv('emea_no_uk.csv', index=False, header=True)

