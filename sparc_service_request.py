#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('dataset.csv')


# In[3]:


dataset.head()


# In[4]:


ticket_counts = dataset.groupby('gilead_location_level2').size().reset_index(name='tickets')

top_5_countries = ticket_counts.sort_values(by='tickets', ascending=False).head(10)
print(top_5_countries)


# In[5]:


region_country_summary = dataset.groupby(['gilead_location_level1', 'gilead_location_level2'])['unique_requests'].sum().reset_index()

top_countries_per_region = region_country_summary.sort_values(['gilead_location_level1', 'unique_requests'], ascending=[True, False]).groupby('gilead_location_level1').head(1)

print(top_countries_per_region)


# In[6]:


location_requests = dataset.groupby('gilead_location_level1')['unique_requests'].sum()

plt.figure(figsize=(10, 7))  
location_requests.plot(kind='bar', color='pink')
plt.title('Unique Requests per Location')
plt.xlabel('Location')
plt.ylabel('Unique Requests')
plt.xticks(rotation=35)  
plt.tight_layout()
plt.grid(True)
plt.show()


# In[7]:


dataset['sys_created_date'] = pd.to_datetime(dataset['sys_created_date'])
dataset['month_year'] = dataset['sys_created_date'].dt.to_period('M')
monthly_requests = dataset.groupby('month_year')['unique_requests'].sum()

#Plotting
plt.figure(figsize=(10, 6))
monthly_requests.plot(kind='line', marker='o', color='blue')
plt.title('Total Unique Requests per Month')
plt.xlabel('Month')
plt.ylabel('Total Unique Requests')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


dataset['sys_created_date'] = pd.to_datetime(dataset['sys_created_date'])
dataset['weekday'] = dataset['sys_created_date'].dt.day_name()
dataset['month_year'] = pd.to_datetime(dataset['sys_created_date']).dt.to_period('M')  

weekday_avg_requests = dataset.groupby(['month_year', 'weekday'])['unique_requests'].mean().unstack()
weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_avg_requests = weekday_avg_requests[weekdays_order]

#Plotting
plt.figure(figsize=(10, 6))
weekday_avg_requests.plot(kind='bar', width=0.8)
plt.title('Average Unique Requests per Weekday for Each Month')
plt.xlabel('Month')
plt.ylabel('Average Unique Requests')
plt.legend(title='Weekday')
plt.grid(True)
plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


df = pd.read_csv('starburstexportday.csv')


# In[11]:


df.head()


# In[12]:


ticket_counts = df.groupby('day_name')['unique_tickets'].sum()

# Plotting
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ticket_counts.plot(kind='bar', color='pink')
plt.title('Number of Unique Tickets by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Unique Tickets')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[13]:


ticket_counts = df.groupby(['created_year', 'day_name'])['unique_tickets'].sum().reset_index()
pivot_table = ticket_counts.pivot(index='day_name', columns='created_year', values='unique_tickets')

# Plotting
plt.figure(figsize=(12, 6))  # Adjust figure size if needed
pivot_table.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Number of Unique Tickets by Day of the Week and Year')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Unique Tickets')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(title='Year')
plt.tight_layout()
plt.show()



# In[14]:


ticket_counts = df.groupby(['gilead_location_level1', 'day_name'])['unique_tickets'].sum().reset_index()
pivot_table = ticket_counts.pivot(index='day_name', columns='gilead_location_level1', values='unique_tickets')

# Plotting
plt.figure(figsize=(12, 6))
pivot_table.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Number of Unique Tickets by Day of the Week and Location')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Unique Tickets')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(title='Location')
plt.tight_layout()
plt.show()


# In[15]:


monthly_ticket_counts = df.groupby('created_month')['unique_tickets'].sum()

# Plotting
plt.figure(figsize=(10, 6)) 
plt.plot(monthly_ticket_counts.index, monthly_ticket_counts.values, marker='o', linestyle='-')
plt.title('Monthly Trends of Unique Tickets')
plt.xlabel('Month')
plt.ylabel('Total Number of Unique Tickets')
plt.xticks(monthly_ticket_counts.index)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[16]:


import seaborn as sns


# In[17]:


ticket_counts = df.groupby(['created_year', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='created_year', y='unique_tickets', hue='gilead_location_level2', data=ticket_counts)
plt.title('Unique Tickets by Year and Location Level 2')
plt.xlabel('Year')
plt.ylabel('Total Number of Unique Tickets')
plt.grid(True)
plt.tight_layout()
plt.legend(title='Location Level 2')
plt.show()


# In[18]:


apac_data = df[df['gilead_location_level1'] == 'APAC (Geographic)']
top_countries = apac_data['gilead_location_level2'].value_counts().head(5)
print("Top 5 countries in APAC (Geographic):", top_countries)


# In[19]:


apac_data = df[df['gilead_location_level1'] == 'EMEA (Geographic)']
top_countries = apac_data['gilead_location_level2'].value_counts().head(5)
print("Top 5 countries in EMEA:", top_countries)


# In[20]:


apac_data = df[df['gilead_location_level1'] == 'LATAM (Geographic)']
top_countries = apac_data['gilead_location_level2'].value_counts().head(5)
print("Top 5 countries in LATAM:", top_countries)


# In[21]:


apac_data = df[df['gilead_location_level1'] == 'North America (Geographic)']
top_countries = apac_data['gilead_location_level2'].value_counts().head(5)
print("Top 5 countries in North America:", top_countries)


# In[22]:


apac_data = df[df['gilead_location_level1'] == 'Unspecified']
top_countries = apac_data['gilead_location_level2'].value_counts().head(5)
print("Unspecified countries", top_countries)


# In[23]:


apac_data = df[df['gilead_location_level1'] == 'APAC (Geographic)']

ticket_counts = apac_data.groupby('gilead_location_level2')['unique_tickets'].sum()
top_countries = ticket_counts.sort_values(ascending=False).head(5)

# Plotting
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='pink')
plt.title('Top 5 Countries in APAC by Ticket Count')
plt.xlabel('APAC Countries')
plt.ylabel('Ticket Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


emea_data = df[df['gilead_location_level1'] == 'EMEA (Geographic)']

ticket_counts = emea_data.groupby('gilead_location_level2')['unique_tickets'].sum()
top_countries = ticket_counts.sort_values(ascending=False).head(5)

# Plotting
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='pink')
plt.title('Top 5 Countries in EMEA by Ticket Count')
plt.xlabel('EMEA Countries')
plt.ylabel('Ticket Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


latam_data = df[df['gilead_location_level1'] == 'LATAM (Geographic)']

ticket_counts = latam_data.groupby('gilead_location_level2')['unique_tickets'].sum()
top_countries = ticket_counts.sort_values(ascending=False).head(5)

# Plotting
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='pink')
plt.title('Top 5 Countries in LATAM by Ticket Count')
plt.xlabel('LATAM Countries')
plt.ylabel('Ticket Count')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[26]:


northamerica_data = df[df['gilead_location_level1'] == 'North America (Geographic)']

ticket_counts = northamerica_data.groupby('gilead_location_level2')['unique_tickets'].sum()
top_countries = ticket_counts.sort_values(ascending=False).head(5)

# Plotting
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='pink')
plt.title('Top 5 Countries in North America by Ticket Count')
plt.xlabel('North America Countries')
plt.ylabel('Ticket Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


apac_data = df[df['gilead_location_level1'] == 'APAC (Geographic)']
tickets_by_day_country = apac_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)
days = tickets_by_day_country['day_name'].unique()

# Plotting
plt.figure(figsize=(12, 10))

for i, day in enumerate(days):
    plt.subplot(3, 3, i+1)
    day_data = tickets_by_day_country[tickets_by_day_country['day_name'] == day].nlargest(5, 'unique_tickets')
    plt.bar(day_data['gilead_location_level2'], day_data['unique_tickets'], color=plt.cm.Paired(i / len(days)), alpha=0.8)
    plt.title(day)
    plt.xlabel('Country')
    plt.ylabel('Total Tickets')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.ylim(0, tickets_by_day_country['unique_tickets'].max() * 1.1)  
    plt.tight_layout()

plt.suptitle('Top 5 Countries in APAC by Ticket Count for Each Day', y=1.02)
plt.tight_layout()
plt.show()


# In[28]:


latam_data = df[df['gilead_location_level1'] == 'LATAM (Geographic)']
tickets_by_day_country = latam_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
days = tickets_by_day_country['day_name'].unique()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)

# Plotting
plt.figure(figsize=(12, 10))

for i, day in enumerate(days):
    plt.subplot(3, 3, i+1)
    day_data = tickets_by_day_country[tickets_by_day_country['day_name'] == day].nlargest(5, 'unique_tickets')
    plt.bar(day_data['gilead_location_level2'], day_data['unique_tickets'], color=plt.cm.Paired(i / len(days)), alpha=0.8)
    plt.title(day)
    plt.xlabel('Country')
    plt.ylabel('Total Tickets')
    plt.xticks(rotation=45)
    plt.ylim(0, tickets_by_day_country['unique_tickets'].max() * 1.1)  
    plt.grid(True)
    plt.tight_layout()

plt.suptitle('Top 5 Countries in LATAM by Ticket Count for Each Day', y=1.02)
plt.tight_layout()
plt.show()


# In[29]:


emea_data = df[df['gilead_location_level1'] == 'EMEA (Geographic)']
tickets_by_day_country = emea_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
days = tickets_by_day_country['day_name'].unique()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)

# Plotting
plt.figure(figsize=(12, 10))

for i, day in enumerate(days):
    plt.subplot(3, 3, i+1)
    day_data = tickets_by_day_country[tickets_by_day_country['day_name'] == day].nlargest(5, 'unique_tickets')
    plt.bar(day_data['gilead_location_level2'], day_data['unique_tickets'], color=plt.cm.Paired(i / len(days)), alpha=0.8)
    plt.title(day)
    plt.xlabel('Country')
    plt.ylabel('Total Tickets')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.ylim(0, tickets_by_day_country['unique_tickets'].max() * 1.1)  

    plt.tight_layout()

plt.suptitle('Top 5 Countries in EMEA by Ticket Count for Each Day', y=1.02)
plt.tight_layout()
plt.show()


# In[30]:


northamerica_data = df[df['gilead_location_level1'] == 'North America (Geographic)']
tickets_by_day_country = northamerica_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
days = tickets_by_day_country['day_name'].unique()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)

# Plotting
plt.figure(figsize=(12, 10))

for i, day in enumerate(days):
    plt.subplot(3, 3, i+1)
    day_data = tickets_by_day_country[tickets_by_day_country['day_name'] == day].nlargest(5, 'unique_tickets')
    plt.bar(day_data['gilead_location_level2'], day_data['unique_tickets'], color=plt.cm.Paired(i / len(days)), alpha=0.8)
    plt.title(day)
    plt.xlabel('Country')
    plt.ylabel('Total Tickets')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.ylim(0, tickets_by_day_country['unique_tickets'].max() * 1.1)  

    plt.tight_layout()

plt.suptitle('Top 5 Countries in North America by Ticket Count for Each Day', y=1.02)
plt.tight_layout()
plt.show()


# In[31]:


apac_data = df[df['gilead_location_level1'] == 'APAC (Geographic)']
tickets_by_day_country = apac_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)

# Plotting
pivot_table = top_countries_by_day.pivot(index='day_name', columns='gilead_location_level2', values='unique_tickets').fillna(0)
plt.figure(figsize=(12, 8))
colors = plt.cm.Paired(range(pivot_table.shape[1]))

bottoms = pd.Series([0] * len(pivot_table), index=pivot_table.index)
for i, country in enumerate(pivot_table.columns):
    plt.bar(pivot_table.index, pivot_table[country], bottom=bottoms, label=country, color=colors[i], alpha=0.8)
    bottoms += pivot_table[country]

plt.xlabel('Day of Week')
plt.ylabel('Total Tickets')
plt.title('Top 5 Countries in APAC by Ticket Count for Each Day')
plt.xticks(rotation=45)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[32]:


apac_data = df[df['gilead_location_level1'] == 'APAC (Geographic)']
tickets_by_day_country = apac_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)
pivot_table = top_countries_by_day.pivot(index='day_name', columns='gilead_location_level2', values='unique_tickets').fillna(0)

# Plotting
colors = [(0.1, 0.2, 0.5),
          (0.2, 0.5, 0.1),
          (0.7, 0.1, 0.1),
          (0.5, 0.2, 0.5),
          (0.8, 0.6, 0.2)]  
plt.figure(figsize=(14, 12))
bar_width = 0.15
opacity = 0.8
index = range(len(pivot_table.index))

colors = plt.cm.Paired(range(pivot_table.shape[1]))

for i, country in enumerate(pivot_table.columns):
    plt.bar([p + bar_width*i for p in index], pivot_table[country], bar_width,
            alpha=opacity,
            color=colors[i % len(colors)],
            label=country)

plt.xlabel('Day of Week')
plt.ylabel('Total Tickets')
plt.title('Top 5 Countries in APAC by tickets for Each Day')
plt.xticks(index, pivot_table.index)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[33]:


latam_data = df[df['gilead_location_level1'] == 'LATAM (Geographic)']

tickets_by_day_country = latam_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)
pivot_table = top_countries_by_day.pivot(index='day_name', columns='gilead_location_level2', values='unique_tickets').fillna(0)

# Plotting
colors = [(0.1, 0.2, 0.5),
          (0.2, 0.5, 0.1),
          (0.7, 0.1, 0.1),
          (0.5, 0.2, 0.5),
          (0.8, 0.6, 0.2)]  
plt.figure(figsize=(14, 12))
bar_width = 0.20
opacity = 0.8
index = range(len(pivot_table.index))

colors = plt.cm.Paired(range(pivot_table.shape[1]))

for i, country in enumerate(pivot_table.columns):
    plt.bar([p + bar_width*i for p in index], pivot_table[country], bar_width,
            alpha=opacity,
            color=colors[i % len(colors)],
            label=country)

plt.xlabel('Day of Week')
plt.ylabel('Total Tickets')
plt.title('Top 5 Countries in LATAM by tickets for Each Day')
plt.xticks(index, pivot_table.index)
plt.grid(True)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[34]:


northamerica_data = df[df['gilead_location_level1'] == 'North America (Geographic)']

tickets_by_day_country = northamerica_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)
pivot_table = top_countries_by_day.pivot(index='day_name', columns='gilead_location_level2', values='unique_tickets').fillna(0)

# Plotting
colors = [(0.1, 0.2, 0.5),
          (0.2, 0.5, 0.1),
          (0.7, 0.1, 0.1),
          (0.5, 0.2, 0.5),
          (0.8, 0.6, 0.2)]  
plt.figure(figsize=(14, 12))
bar_width = 0.30
opacity = 0.8
index = range(len(pivot_table.index))

colors = plt.cm.Paired(range(pivot_table.shape[1]))

for i, country in enumerate(pivot_table.columns):
    plt.bar([p + bar_width*i for p in index], pivot_table[country], bar_width,
            alpha=opacity,
            color=colors[i % len(colors)],
            label=country)

plt.xlabel('Day of Week')
plt.ylabel('Total Tickets')
plt.title('Top 5 Countries in North America by tickets for Each Day')
plt.xticks(index, pivot_table.index)
plt.grid(True)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[35]:


emea_data = df[df['gilead_location_level1'] == 'EMEA (Geographic)']
tickets_by_day_country = emea_data.groupby(['day_name', 'gilead_location_level2'])['unique_tickets'].sum().reset_index()
days = tickets_by_day_country['day_name'].unique()
top_countries_by_day = tickets_by_day_country.groupby('day_name').apply(lambda x: x.nlargest(5, 'unique_tickets')).reset_index(drop=True)

# Plotting
pivot_table = top_countries_by_day.pivot(index='day_name', columns='gilead_location_level2', values='unique_tickets').fillna(0)
plt.figure(figsize=(19, 15))
bar_width = 0.12
opacity = 0.8
index = range(len(pivot_table.index))

colors = plt.cm.Paired(range(pivot_table.shape[1]))

for i, country in enumerate(pivot_table.columns):
    plt.bar([p + bar_width*i for p in index], pivot_table[country], bar_width,
            alpha=opacity,
            color=colors[i % len(colors)],
            label=country)

plt.xlabel('Day of Week')
plt.ylabel('Total Tickets')
plt.title('Top 5 Countries in EMEA by Ticket Count for Each Day')
plt.xticks(index, pivot_table.index)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

