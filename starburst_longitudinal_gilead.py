#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %pip install colorama
# %pip install progressbar
# %pip install statsmodels
# %pip install trino


# In[3]:


# dbutils.library.restartPython()


# In[4]:


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


# In[5]:


actuals = pd.read_csv('starburstlongitudinal.csv')


# In[6]:


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


# In[7]:


# #sb dev connection test
# conn = trino.dbapi.connect(
#     host='query.gilead.com',
#     port=443,
#     http_scheme='https',
#     user='divya.shah15@gilead.com',
#     auth=OAuth2Authentication(),
# )


# In[8]:


# cur = conn.cursor()
# # cur.execute('select unq.sys_created_date,unq.created_year,unq.created_month,unq.created_week,unq.created_day_number,unq."organization_flag",sum(unq.sparc_ticket_date_order) as unique_tickets from(select *,row_number() over(partition by sr_number order by sys_created_date asc) as sparc_ticket_date_order from(select distinct sys_created_date,"gilead_location_level1","gilead_location_level2","organization_flag",extract(year from sys_created_date) as created_year,extract(month from sys_created_date) as created_month,extract(week from sys_created_date) as created_week,extract(day_of_week from sys_created_date) as created_day_number,sr_number from "gna_it_services"."edp-gna-itservices-sparc"."sparc_service_request_final" where extract(year from sys_created_date) >= 2022 and sys_created_date < date \'2024-06-01\') as sq) as unq where unq.sparc_ticket_date_order = 1 group by unq.sys_created_date,unq.created_year,unq.created_month,unq.created_week,unq.created_day_number,unq."organization_flag" order by unq.sys_created_date asc,unq."organization_flag" asc')
# cur.execute("select * from gna_it_services.edp-gna-itservices-sparc.sparc_service_request_final")
# rows = fetch_rows(cur)
# columns = [x[0] for x in cur.description]
# actuals = pd.DataFrame(rows, columns=columns)
# row_count = len(actuals)
# print(Fore.BLUE, f'Query returned {row_count} rows')
# print(Fore.YELLOW,actuals.columns)
# display(actuals)


# In[9]:


df = actuals.copy()
df = df.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
df = df[(df.organization_flag == 'Gilead')]
df.head()


# In[10]:


preserve_columns = ['ds', 'y']
df = df.filter(items = preserve_columns)
df.head()
df.dtypes


# In[11]:


df['ds'] = pd.DatetimeIndex(df['ds'])
df.dtypes
df = df.set_index(pd.DatetimeIndex(df['ds'])).drop('ds', axis=1)


# In[12]:


df.plot()


# In[13]:


print(df.max(), df.idxmax())


# In[14]:


df.loc['2023-09-12', 'y']


# In[15]:


df.loc['2023-09-11', 'y'] = 1269
df.plot()


# In[16]:


df_weekdays = df.copy()
df_weekdays['day_name'] = df.index.day_name()
print(df_weekdays.iloc[100:125])


# In[17]:


df.resample('W').sum().plot()


# In[18]:


df.resample('M').sum().plot()


# In[19]:


df_roll = df.copy()
df_roll['30_day_ma'] = df_roll.y.rolling(30, center=True).mean()
df_roll.plot()


# In[20]:


df_roll['30_std'] = df_roll.y.rolling(30, center=True).std()
df_roll.plot()


# In[21]:


df['day'] = df.index.dayofweek
df['week'] = df.index.strftime('%U').astype(int)
seasonal_plot(df, y='y', period='week', freq='day')


# In[22]:


df['dayofyear'] = df.index.dayofyear
df['year'] = df.index.year
seasonal_plot(df, y='y', period='year', freq='dayofyear')


# In[23]:


plot_periodogram(df.y)


# In[24]:


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq='A', order=10)


dp = DeterministicProcess(
    index=df.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)

X=dp.in_sample()
X.head()


# In[26]:


y = df['y']
model = LinearRegression(fit_intercept=False)
_ = model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=365)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax=y.plot(color='0.25', style='.', title='Seasonal Forecast')
ax=y_pred.plot(ax=ax, label='Seasonal')
ax=y_fore.plot(ax=ax, label='Forecast')
_ = ax.legend


# In[27]:


prep_df = actuals.copy()
prep_df = prep_df.rename(columns={'sys_created_date':'ds', 'unique_tickets':'y'})
prep_df = prep_df[(prep_df.organization_flag == 'Gilead')]
preserve_columns = ['ds', 'y']
prep_df = prep_df.filter(items = preserve_columns)
prep_df['ds'] = pd.DatetimeIndex(prep_df['ds'])
prep_df.reset_index(drop=True)
df = prep_df.copy()
 
m = Prophet(
    daily_seasonality = False,
    weekly_seasonality = True,
    yearly_seasonality = True,
    changepoint_prior_scale = 0.05,
    seasonality_prior_scale = 10,
    seasonality_mode = 'additive',
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

