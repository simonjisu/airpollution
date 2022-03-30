#!/usr/bin/env python
# coding: utf-8

# # Data Handling

# In[1]:


import sys
import numpy as np
import pandas as pd
# from datetime import datetime as dt
# from datetime import timedelta
from tqdm.notebook import tqdm
from pathlib import Path

proj_path = Path().absolute().parent
sys.path.append(str(proj_path))
data_path = proj_path / 'data' 

from src.dbengine import DBEngine
db = DBEngine(db_path=data_path / "airpollution.db")


# In[2]:


# what columns to use?
columns = ['district', 'datetime', 'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']

sql = """
SELECT aq.measure_code, aq.datetime, aq.SO2, aq.CO, aq.O3, aq.NO2, aq.PM10, aq.PM25
FROM airquality AS aq
"""
res = db.query(sql)
df = pd.DataFrame(res).rename(columns=dict(enumerate(columns)))
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index(['district', 'datetime']).sort_index()
# should fill the null values using previous datetime's value

df = df.mask(df.isnull()).groupby([df.index.get_level_values(1).time]).fillna(method = 'ffill')
df.isnull().sum()


# In[3]:


# what columns to use?
# resample 

columns = [
    'datetime', 'PM10', 'PM25', 'SO2', 'CO', 'O3', 'NO2', 'temperature', 
    'precipitation', 'wind_speed', 'wind_direction', 'humidity', 'vapor_pressure', 
    'local_pressure', 'sea_level_pressure', 'sunshine', 'solar_radiation', 
    'ground_temperature', '5cm_soil_temperature', '10cm_soil_temperature', '20cm_soil_temperature', '30cm_soil_temperature'
]

sql = """
SELECT 
    a.datetime, a.PM10, a.PM25, a.SO2, a.CO, a.O3, a.NO2, w.temperature,
    w.precipitation, w.wind_speed, w.wind_direction, w.humidity, w.vapor_pressure, 
    w.local_pressure, w.sea_level_pressure, w.sunshine, w.solar_radiation,
    w."ground_temperature", w."5cm_soil_temperature", w."10cm_soil_temperature", w."20cm_soil_temperature", w."30cm_soil_temperature"
FROM (
    SELECT 
        aq.datetime, AVG(aq.SO2) SO2, AVG(aq.CO) CO, AVG(aq.O3) O3, AVG(aq.NO2) NO2, AVG(aq.PM10) PM10, AVG(aq.PM25) PM25
    FROM airquality AS aq
    GROUP BY aq.datetime
    ORDER BY aq.datetime
) AS a
JOIN weather AS w
ON a.datetime = w.datetime
"""
res = db.query(sql)
df = pd.DataFrame(res).rename(columns=dict(enumerate(columns))).set_index(['datetime']).sort_index()


# In[4]:


df_null = len(df) - df.describe().loc['count']
print(df_null.index[(df_null > 0).values].values)
df_null


# In[5]:


check_columns = [
    'temperature', 'wind_speed', 'wind_direction', 'vapor_pressure', 'local_pressure', 'sea_level_pressure',
    'ground_temperature', '5cm_soil_temperature', '10cm_soil_temperature', '20cm_soil_temperature', '30cm_soil_temperature'
]

for c in check_columns:
    idx = df.loc[df[c].isnull(), c].index
    print(c)
    if len(idx) > 4:
        print(idx.values[:2], '...', idx.values[-2:])
    else:
        print(idx.values)


# In[ ]:




