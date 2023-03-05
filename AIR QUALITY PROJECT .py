#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import folium


# In[3]:


AQ = pd.read_csv('final_data.csv');AQ


# In[4]:


AQ.shape


# In[5]:


AQ.info()


# In[6]:


AQ[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO','O3']] = AQ[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO','O3']].replace(0, pd.NaT)


# In[7]:


AQ.fillna(AQ.mean(),inplace=True)


# In[8]:


AQ.rename(columns={'Unnamed: 0': 'sr'},inplace=True)


# In[9]:


AQ.drop('sr',axis=1,inplace=True)


# In[11]:


AQ['Date'] = pd.to_datetime(AQ['Date'],format='%Y-%M-%D')


# In[12]:


def aqi(pollutant, conc):
    # Define the AQI breakpoints and corresponding AQI values
    breakpoints = [0, 50, 100, 168, 208, 748, 1000]
    aqi_values = [0, 50, 100, 150, 200, 300, 400]

    # Calculate the AQI
    i = 0
    while i < len(breakpoints) - 1 and conc > breakpoints[i+1]:
        i += 1
    aqi = ((aqi_values[i+1] - aqi_values[i]) / (breakpoints[i+1] - breakpoints[i])) * (conc - breakpoints[i]) + aqi_values[i]
    return aqi


# In[13]:


AQ['PM2.5_AQI'] = AQ['PM2.5'].apply(lambda x: aqi('PM2.5', x))
AQ['PM10_AQI'] = AQ['PM10'].apply(lambda x: aqi('PM10', x))
AQ['NO2_AQI'] = AQ['NO2'].apply(lambda x: aqi('NO2', x))
AQ['SO2_AQI'] = AQ['SO2'].apply(lambda x: aqi('SO2', x))
AQ['CO_AQI'] = AQ['CO'].apply(lambda x: aqi('CO', x))
AQ['O3_AQI'] = AQ['O3'].apply(lambda x: aqi('O3', x))


# In[14]:


AQ['AQI'] = 0.0
for pollutant in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO','O3']:
    AQ['AQI'] = AQ.apply(lambda row: aqi(pollutant, row[pollutant]), axis=1)


# In[15]:


AQ.drop(['PM2.5_AQI','PM10_AQI','NO2_AQI','SO2_AQI','CO_AQI','O3_AQI'],axis=1,inplace=True)


# In[16]:


AQ.to_csv('AQFD.csv')


# In[17]:


sns.heatmap(AQ.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[18]:


AQ.describe()


# In[19]:


# Calculate the daily average for each pollutant
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
for pollutant in pollutants:
    AQ[pollutant+'_daily_avg'] = AQ[pollutant].groupby(AQ['Date']).transform('mean')

# Plot the daily averages
for pollutant in pollutants:
    plt.figure()
    AQ.plot(x='Date', y=pollutant+'_daily_avg', figsize=(10,5))
    plt.title('Daily Average ' + pollutant + ' Concentration')
    plt.xlabel('year')
    plt.ylabel(pollutant + ' Concentration')


# In[54]:


# Calculate the correlation between each pair of pollutants
corr = AQ[pollutants].corr()

# Plot the correlation matrix
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Pollutant Correlation Matrix')


# In[57]:


# Calculate the monthly average for each pollutant
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
for pollutant in pollutants:
     AQ[pollutant+'_monthly_avg'] = AQ[pollutant].groupby(AQ['Date'].dt.to_period('M')).transform('mean')


# Plot the monthly averages
for pollutant in pollutants:
    plt.figure()
    AQ.plot(x='Date', y=pollutant+'_monthly_avg', figsize=(10,5))
    plt.title('Monthly Average ' + pollutant + ' Concentration')
    plt.xlabel('year')
    plt.ylabel(pollutant + ' Concentration')


# In[58]:


grouped = AQ.groupby(['Date']).agg({'PM2.5': 'mean', 'PM10': 'mean', 'SO2': 'mean', 'NO2': 'mean', 'CO': 'mean','O3': 'mean'}).reset_index()


# In[59]:


grouped = grouped.set_index('Date')
grouped = grouped.resample('D').mean()


# In[60]:


plt.figure(figsize=(10,9))
plt.plot(grouped.index, grouped['PM2.5'], label='PM2.5')
plt.plot(grouped.index, grouped['PM10'], label='PM10')
plt.plot(grouped.index, grouped['SO2'], label='SO2')
plt.plot(grouped.index, grouped['NO2'], label='NO2')
plt.plot(grouped.index, grouped['CO'], label='CO')
plt.plot(grouped.index, grouped['O3'], label='O3')
plt.legend()
plt.xlabel('year')
plt.ylabel(pollutant + ' Concentration')
plt.show()


# In[64]:


# Plot PM2.5 time series with 7-day rolling average
plt.figure(figsize=(10,5))
plt.plot(AQ['Date'], AQ['PM2.5'])
plt.plot(AQ['Date'], AQ['PM2.5'].rolling(7).mean())
plt.title('PM2.5 Time Series with 7-Day Rolling Average')
plt.xlabel('year')
plt.ylabel('PM2.5 Concentration')
plt.legend(['PM2.5 Concentration','PM2.5 Concentration with 7-Day Rolling Average'])
plt.show()


# In[65]:


# Convert the 'Date' column to numeric values
AQ['Numeric_Date'] = pd.to_numeric(AQ['Date'].apply(pd.Timestamp.timestamp))

# Plot the time series with a trend line
plt.plot(AQ['Date'], AQ['PM2.5'])
plt.plot(AQ['Date'], np.poly1d(np.polyfit(AQ['Numeric_Date'], AQ['PM2.5'], 1))(AQ['Numeric_Date']))
plt.xlabel('year')
plt.ylabel('PM2.5 Concentration')
plt.legend(['PM2.5 Concentration','trend line'])
plt.show()


# In[66]:


# Plot PM10 time series with 7-day rolling average
plt.figure(figsize=(10,5))
plt.plot(AQ['Date'], AQ['PM10'])
plt.plot(AQ['Date'], AQ['PM10'].rolling(7).mean())
plt.title('PM10 Time Series with 7-Day Rolling Average')
plt.xlabel('year')
plt.ylabel('PM10 Concentration')
plt.legend(['PM10 Concentration','PM10 Concentration with 7-Day Rolling Average'])
plt.show()


# In[67]:


# Plot the time series with a trend line
plt.plot(AQ['Date'], AQ['PM10'])
plt.plot(AQ['Date'], np.poly1d(np.polyfit(AQ['Numeric_Date'], AQ['PM10'], 1))(AQ['Numeric_Date']))
plt.xlabel('year')
plt.ylabel('PM10 Concentration')
plt.legend(['PM10 Concentration','trend line'])
plt.show()


# In[69]:


# Plot NO2 time series with 7-day rolling average
plt.figure(figsize=(10,5))
plt.plot(AQ['Date'], AQ['NO2'])
plt.plot(AQ['Date'], AQ['NO2'].rolling(7).mean())
plt.title('PM10 Time Series with 7-Day Rolling Average')
plt.xlabel('year')
plt.ylabel('NO2 Concentration')
plt.legend(['NO2 Concentration','NO2 Concentration with 7-Day Rolling Average'])
plt.show()


# In[44]:


# Plot the time series with a trend line
plt.plot(AQ['Date'], AQ['NO2'])
plt.plot(AQ['Date'], np.poly1d(np.polyfit(AQ['Numeric_Date'], AQ['NO2'], 1))(AQ['Numeric_Date']))
plt.xlabel('year')
plt.ylabel('NO2 Concentration')
plt.legend(['NO2 Concentration','trend line'])
plt.show()


# In[70]:


# Plot CO time series with 7-day rolling average
plt.figure(figsize=(10,5))
plt.plot(AQ['Date'], AQ['CO'])
plt.plot(AQ['Date'], AQ['CO'].rolling(7).mean())
plt.title('CO Time Series with 7-Day Rolling Average')
plt.xlabel('year')
plt.ylabel('CO Concentration')
plt.legend(['CO Concentration','CO Concentration with 7-Day Rolling Average'])
plt.show()


# In[71]:


# Plot the time series with a trend line
plt.plot(AQ['Date'], AQ['CO'])
plt.plot(AQ['Date'], np.poly1d(np.polyfit(AQ['Numeric_Date'], AQ['CO'], 1))(AQ['Numeric_Date']))
plt.xlabel('year')
plt.ylabel('CO Concentration')
plt.legend(['CO Concentration','trend line'])
plt.show()


# In[72]:


# Plot CO time series with 7-day rolling average
plt.figure(figsize=(10,5))
plt.plot(AQ['Date'], AQ['SO2'])
plt.plot(AQ['Date'], AQ['SO2'].rolling(7).mean())
plt.title('SO2 Time Series with 7-Day Rolling Average')
plt.xlabel('year')
plt.ylabel('SO2 Concentration')
plt.legend(['SO2 Concentration','SO2 Concentration with 7-Day Rolling Average'])
plt.show()


# In[73]:


# Plot the time series with a trend line
plt.plot(AQ['Date'], AQ['SO2'])
plt.plot(AQ['Date'], np.poly1d(np.polyfit(AQ['Numeric_Date'], AQ['SO2'], 1))(AQ['Numeric_Date']))
plt.xlabel('year')
plt.ylabel('SO2 Concentration')
plt.legend(['SO2 Concentration','trend line'])
plt.show()


# In[74]:


# Plot CO time series with 7-day rolling average
plt.figure(figsize=(10,5))
plt.plot(AQ['Date'], AQ['O3'])
plt.plot(AQ['Date'], AQ['O3'].rolling(7).mean())
plt.title('O3 Time Series with 7-Day Rolling Average')
plt.xlabel('year')
plt.ylabel('O3 Concentration')
plt.legend(['O3 Concentration','O3 Concentration with 7-Day Rolling Average'])
plt.show()


# In[75]:


# Plot the time series with a trend line
plt.plot(AQ['Date'], AQ['O3'])
plt.plot(AQ['Date'], np.poly1d(np.polyfit(AQ['Numeric_Date'], AQ['O3'], 1))(AQ['Numeric_Date']))
plt.xlabel('year')
plt.ylabel('O3 Concentration')
plt.legend(['O3 Concentration','trend line'])
plt.show()


# In[19]:


# Identify and analyze the levels and trends of air pollutants in a specific area, and identify sources of pollution.

# Plot the trend of each pollutant over time
AQ.plot(x='Date', y=['NO2', 'PM10', 'SO2', 'CO'])
plt.xlabel('Date')
plt.ylabel('Concentration (µg/m³)')
plt.title('Trends in Air Pollutants')
plt.show()


# In[64]:


# Identify the sources of pollution using a bar chart
AQ[['PM2.5', 'PM10', 'NO2', 'SO2','CO','O3']].sum().plot(kind='bar')
plt.xlabel('Source of Pollution')
plt.ylabel('Concentration (µg/m³)')
plt.title('Sources of Air Pollution')
plt.show()


# In[38]:


#bar plot of PM2.5 vs City - desc order
AQ[['PM2.5', 'City']].groupby(['City']).median().sort_values("PM2.5", ascending = False).plot.bar()


# In[40]:


#bar plot of PM10 vs City - desc order
AQ[['PM10', 'City']].groupby(['City']).median().sort_values("PM10", ascending = False).plot.bar()


# In[41]:


#bar plot of NO2 vs City - desc order
AQ[['NO2', 'City']].groupby(['City']).median().sort_values("NO2", ascending = False).plot.bar()


# In[42]:


#bar plot of CO vs City - desc order
AQ[['CO', 'City']].groupby(['City']).median().sort_values("CO", ascending = False).plot.bar()


# In[43]:


#bar plot of SO2 vs City - desc order
AQ[['SO2', 'City']].groupby(['City']).median().sort_values("SO2", ascending = False).plot.bar()


# In[44]:


#bar plot of O3 vs City - desc order
AQ[['O3', 'City']].groupby(['City']).median().sort_values("O3", ascending = False).plot.bar()


# In[48]:


#Scatter plots of all columns
sns.set()
cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2','O3']
sns.pairplot(AQ[cols],size = 2.5)
plt.show()


# In[51]:


#Correlation matrix
corrmat = AQ.corr()
f, ax = plt.subplots(figsize = (6, 5))
sns.heatmap(corrmat, vmax = 1, square = True, annot = True)


# In[54]:


AQ['year'] = AQ['Date'].dt.year # year
AQ['year'] = AQ['year'].fillna(0.0).astype(int)
AQ = AQ[(AQ['year']>0)]

AQ.head()


# In[57]:


# Heatmap Pivot with City as Row, Year as Col, PM2.5 as Value
f, ax = plt.subplots(figsize = (10,10))
ax.set_title('{} by City and year'.format('PM2.5'))
sns.heatmap(AQ.pivot_table('PM2.5', index = 'City',
                columns = ['year'], aggfunc = 'median', margins=True),
                annot = True, cmap = 'YlGnBu', linewidths = 1, ax = ax, cbar_kws = {'label': 'Average taken Annually'})


# In[58]:


# Heatmap Pivot with City as Row, Year as Col, PM10 as Value
f, ax = plt.subplots(figsize = (10,10))
ax.set_title('{} by City and year'.format('PM10'))
sns.heatmap(AQ.pivot_table('PM10', index = 'City',
                columns = ['year'], aggfunc = 'median', margins=True),
                annot = True, cmap = 'YlGnBu', linewidths = 1, ax = ax, cbar_kws = {'label': 'Average taken Annually'})


# In[59]:


# Heatmap Pivot with City as Row, Year as Col, NO2 as Value
f, ax = plt.subplots(figsize = (10,10))
ax.set_title('{} by City and year'.format('NO2'))
sns.heatmap(AQ.pivot_table('NO2', index = 'City',
                columns = ['year'], aggfunc = 'median', margins=True),
                annot = True, cmap = 'YlGnBu', linewidths = 1, ax = ax, cbar_kws = {'label': 'Average taken Annually'})


# In[60]:


# Heatmap Pivot with City as Row, Year as Col, CO as Value
f, ax = plt.subplots(figsize = (10,10))
ax.set_title('{} by City and year'.format('CO'))
sns.heatmap(AQ.pivot_table('CO', index = 'City',
                columns = ['year'], aggfunc = 'median', margins=True),
                annot = True, cmap = 'YlGnBu', linewidths = 1, ax = ax, cbar_kws = {'label': 'Average taken Annually'})


# In[79]:


# Heatmap Pivot with City as Row, Year as Col, SO2 as Value
f, ax = plt.subplots(figsize = (10,10))
ax.set_title('{} by City and year'.format('SO2'))
sns.heatmap(AQ.pivot_table('SO2', index = 'City',
                columns = ['year'], aggfunc = 'median', margins=True),
                annot = True, cmap = 'YlGnBu', linewidths = 1, ax = ax, cbar_kws = {'label': 'Average taken Annually'})


# In[80]:


# Heatmap Pivot with City as Row, Year as Col, O3 as Value
f, ax = plt.subplots(figsize = (10,10))
ax.set_title('{} by City and year'.format('O3'))
sns.heatmap(AQ.pivot_table('O3', index = 'City',
                columns = ['year'], aggfunc = 'median', margins=True),
                annot = True, cmap = 'YlGnBu', linewidths = 1, ax = ax, cbar_kws = {'label': 'Average taken Annually'})


# In[19]:


AQ


# In[20]:


# Load the dataset of city locations
city_locations = pd.read_csv("C:\\Users\\ASUS\\Downloads\\updated.csv")

# Merge the city locations with your dataset
aq = pd.merge(AQ, city_locations[['city', 'lat', 'lng']], left_on='City', right_on='city')

# Rename the 'lat' and 'lng' columns for clarity
aq = aq.rename(columns={'lat': 'latitude', 'lng': 'longitude'})

# Preview the updated dataset
print(aq.head())


# In[24]:


aq


# In[21]:


import folium
from folium.plugins import HeatMap

# Set the center of the map to the average latitude and longitude
center_lat = aq['latitude'].mean()
center_lon = aq['longitude'].mean()

# Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=3)

# Create a HeatMap layer for the PM2.5 concentration data
data = aq[['latitude', 'longitude', 'PM2.5']].values.tolist()
HeatMap(data).add_to(m)

# Display the map
m


# In[22]:


# Create a HeatMap layer for the PM10 concentration data
data = aq[['latitude', 'longitude', 'PM10']].values.tolist()
HeatMap(data).add_to(m)

# Display the map
m


# In[23]:


# Create a HeatMap layer for the NO2 concentration data
data = aq[['latitude', 'longitude', 'NO2']].values.tolist()
HeatMap(data).add_to(m)

# Display the map
m


# In[25]:


# Create a HeatMap layer for the CO concentration data
data = aq[['latitude', 'longitude', 'CO']].values.tolist()
HeatMap(data).add_to(m)

# Display the map
m


# In[27]:


# Create a HeatMap layer for the SO2 concentration data
data = aq[['latitude', 'longitude', 'SO2']].values.tolist()
HeatMap(data).add_to(m)

# Display the map
m


# In[28]:


# Create a HeatMap layer for the O3 concentration data
data = aq[['latitude', 'longitude', 'O3']].values.tolist()
HeatMap(data).add_to(m)

# Display the map
m


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




