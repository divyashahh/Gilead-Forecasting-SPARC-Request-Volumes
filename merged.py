#!/usr/bin/env python
# coding: utf-8

# In[69]:


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


# In[70]:


apac_china = pd.read_csv('apac_china.csv')
apac_no_china = pd.read_csv('apac_no_china.csv')
north_us = pd.read_csv('north_us.csv')
north_no_us = pd.read_csv('north_no_us.csv')
emea_no_uk = pd.read_csv('emea_no_uk.csv')
emea_uk = pd.read_csv('emea_uk.csv')
latam_no_brazil = pd.read_csv('latam_no_brazil.csv')
brazil = pd.read_csv('brazil.csv')
netherlands = pd.read_csv('netherlands.csv')
actuals = pd.read_csv('starburstlongitudinal2.csv')


# In[71]:


actuals_filtered = actuals[['sys_created_date', 'unique_tickets']]
actuals_filtered = actuals_filtered.rename(columns={'sys_created_date': 'ds'})


# In[72]:


actuals_filtered = actuals_filtered.groupby(['ds'])['unique_tickets'].sum().reset_index(name ='unique_tickets')
display(actuals_filtered)


# In[73]:


dataframes = [apac_china, apac_no_china, north_us, north_no_us, emea_no_uk, emea_uk, latam_no_brazil, brazil, netherlands,]

for df in dataframes:
    if 'accuracy_by_day' in df.columns:
        df.drop(columns=['accuracy_by_day'], inplace=True)
        
merged_df = actuals_filtered.copy() 
for df in dataframes: 
    merged_df = merged_df.merge(df, on='ds', how='left')
display(merged_df)


# In[74]:


merged_df.columns


# In[75]:


merged_min = merged_df[['ds','unique_tickets','yhat_apac_china','yhat_apac_no_china','yhat_north_us','yhat_north_no_us','yhat_emea_no_uk','yhat_emea_uk','yhat_latam_no_brazil','yhat_brazil.1','yhat_netherlands.1']]
merged_min.columns


# In[76]:


#merged_min.fillna(value={'yhat_apac_no_china': 0}, inplace=True)
merged_min=merged_min.fillna(0)


# In[77]:


display(merged_min)


# In[78]:


merged_min['predictions']= merged_min['yhat_apac_china']+merged_min['yhat_apac_no_china']+merged_min['yhat_north_us']+merged_min['yhat_north_no_us']+merged_min['yhat_emea_no_uk']+merged_min['yhat_emea_uk']+merged_min['yhat_latam_no_brazil']+merged_min['yhat_brazil.1']+merged_min['yhat_netherlands.1']


# In[79]:


display(merged_min)


# In[80]:


merged_min = merged_min[['ds','unique_tickets','predictions']]
display(merged_min)


# In[81]:


merged_min['percent_error'] = (merged_min['unique_tickets'] - merged_min['predictions']) / merged_min['unique_tickets']
display(merged_min)


# In[82]:


merged_min['abs_percent_error'] = abs(merged_min['percent_error'])
display(merged_min)


# In[83]:


mean_abs_percent_error = merged_min['abs_percent_error'].mean()
print("MAPE:", mean_abs_percent_error)


# In[111]:


plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(merged_min['unique_tickets'], label='Unique Tickets', color='blue')
plt.plot(merged_min['predictions'], label='Predictions', color='orange')
plt.title('Unique Tickets & Predictions')
plt.xlabel('Index')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()


# In[106]:


plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 2)
plt.plot(merged_min['abs_percent_error'], label='Abs Percent Error', color='deeppink')
plt.title('Absolute Percent Error')
plt.xlabel('Index')
plt.ylabel('Abs Percent Error')
plt.legend()

plt.tight_layout()
plt.show()


# In[86]:


merged_new = merged_min[['ds','unique_tickets','predictions']]
display(merged_new)


# In[87]:


merged_new['ds'] = pd.to_datetime(merged_new['ds'])
merged_new['year_month'] = merged_new['ds'].dt.to_period('M')
merged_month = merged_new.groupby('year_month')[['unique_tickets', 'predictions']].sum()
merged_month = merged_month.reset_index()
display(merged_month)


# In[88]:


merged_month['percent_error'] = (merged_month['unique_tickets'] - merged_month['predictions']) / merged_month['unique_tickets']
merged_month['abs_percent_error'] = abs(merged_month['percent_error'])
display(merged_month)


# In[89]:


mean_abs_percent_error = merged_month['abs_percent_error'].mean()
print("MAPE:", mean_abs_percent_error)


# In[105]:


plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(merged_month['unique_tickets'], label='Unique Tickets', color='cyan')
plt.plot(merged_month['predictions'], label='Predictions', color='indigo')
plt.title('Unique Tickets & Predictions')
plt.xlabel('Index')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()


# In[103]:


plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 2)
plt.plot(merged_month['abs_percent_error'], label='Abs Percent Error', color='hotpink')
plt.title('Absolute Percent Error')
plt.xlabel('Index')
plt.ylabel('Abs Percent Error')
plt.legend()
plt.tight_layout()
plt.show()


# In[97]:


merged_month.to_csv('merged_month.csv', index=False, header=True)

