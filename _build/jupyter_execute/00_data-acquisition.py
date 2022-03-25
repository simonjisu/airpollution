#!/usr/bin/env python
# coding: utf-8

# # Data Acquisition

# In[1]:


import os
import sqlite3
import pandas as pd
from pathlib import Path

from collections import defaultdict
from tqdm.notebook import tqdm
from typing import Any, Union

proj_path = Path().absolute().parent
data_path = proj_path / 'data' 


# # SQLite3 Database

# In[2]:


conn = sqlite3.connect(data_path / "airpollution.db")


# # Airquality Data
# 
# https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123

# In[3]:


def load_data(path:Path) -> pd.DataFrame:
    """Load data function

    Args:
        path (Path): path of data with file name
        enc (str, optional): encoding. Defaults to 'utf-8'.
    Returns:
        pd.DataFrame 
    """    
    if path.name.split('.')[-1] == 'xlsx':
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='utf-8')

    return df

def filter_seoul(df):
    return df.loc[df['지역'].str.contains('서울'), :]


# In[4]:


datafiles = sorted([x for x in (data_path / 'airquality').glob("*") if x.is_dir()])
# concatnate all files
parts = []
for p_year in tqdm(datafiles, total=len(datafiles)):
    new_path = data_path / 'airquality' / f'air-seoul-{p_year.name}.csv'
    for p in p_year.glob('*'):
        
        df = load_data(p)
        df = filter_seoul(df)
        parts.append(df)
    
    df_all = pd.concat(parts).reset_index(drop=True)
    if p_year.name == '2018':
        # fillna for air-seoul-2018.csv
        # '망' column contains null value due to policy changed
        # create dictionary for measure point
        m_dict = dict(df_all.loc[~df_all['망'].isna(), ['측정소코드', '망']].drop_duplicates().values)
        df_all.loc[df_all['망'].isna(), '망'] = df_all.loc[df_all['망'].isna(), '망'].fillna(df_all['측정소코드'].map(m_dict)).values
    
    df_all.to_csv(new_path, encoding='utf-8', index=False)
    parts = []


# In[33]:


# change column name and insert into database
column_m_dict = {
    '지역': 'district', 
    '측정소코드': 'measurecode', 
    '측정소명': 'measurename', 
    '측정일시': 'date', 
    '주소': 'address',
    '망': 'measurepoint'
}

check_miss_match = {}
for p in sorted((data_path / 'airquality').glob("*.csv")):
    df = pd.read_csv(p, encoding='utf-8').rename(columns=column_m_dict)
    c = df.loc[:, ['district', 'measurepoint', 'measurecode', 'measurename', 'address']].drop_duplicates()
    check_miss_match[int(p.name.rstrip('\.csv').split('-')[-1])] = c
    print(f"{p.name}, num-unique data: {len(c)}, measurecode: {len(c['measurecode'].unique())}, district: {len(c['district'].unique())}, address: {len(c['district'].unique())}")
    # saved changed columns
    df.to_csv(p, encoding='utf-8', index=False)


# In[35]:


# fix the district name and address by 2021 version of measurecode
code2dist = dict(check_miss_match[2021].loc[:, ['measurecode', 'district']].values)
code2add = dict(check_miss_match[2021].loc[:, ['measurecode', 'address']].values)
df = pd.read_csv(data_path / 'airquality' / 'air-seoul-2018.csv', encoding='utf-8').rename(columns=column_m_dict)
df['district'] = df['measurecode'].map(code2dist)
df['address'] = df['measurecode'].map(code2add)
df = df.set_index(['measurecode', 'district', 'measurename', 'address', 'measurepoint']).sort_values(['measurecode', 'date']).reset_index()

# save 
# df.to_csv(data_path / 'airquality' / 'air-seoul-2018.csv', encoding='utf-8', index=False)


# In[4]:


def drop_tables(conn):
    cur = conn.cursor()
    conn.execute("DROP TABLE IF EXISTS airquality;")
    conn.execute("DROP TABLE IF EXISTS airmeasure;")
    cur.close()


# In[24]:


drop_tables(conn)


# In[25]:


cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS airmeasure (
        sid INTEGER PRIMARY KEY,
        measurecode INTEGER NOT NULL UNIQUE,
        district TEXT, 
        measurename TEXT, 
        address TEXT, 
        measurepoint TEXT
    );
    """
)
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS airquality (
        airid INTEGER PRIMARY KEY,
        measurecode INTEGER, 
        date TEXT, 
        SO2 REAL, 
        CO REAL, 
        O3 REAL,
        NO2 REAL, 
        PM10 REAL, 
        PM25 REAL, 
        FOREIGN KEY (measurecode)
            REFERENCES airmeasure (measurecode)
            ON DELETE CASCADE 
            ON UPDATE NO ACTION
    );
    """
)

airmeasure_columns = ['measurecode', 'district', 'measurename', 'address', 'measurepoint']
airquality_columns = ['measurecode', 'date', 'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
df_airmeasure = None

sql_airmeasure = """
INSERT INTO airmeasure (sid, measurecode, district, measurename, address, measurepoint)
VALUES (?, ?, ?, ?, ?, ?);
"""
sql_airquality = """
INSERT INTO airquality (airid, measurecode, date, SO2, CO, O3, NO2, PM10, PM25) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""
idx = 0
for p in tqdm(sorted((data_path / 'airquality').glob("*.csv")), total=4):
    df = pd.read_csv(p, encoding='utf-8')
    df = df.set_index(['measurecode', 'district', 'measurename', 'address', 'measurepoint']).sort_values(['measurecode', 'date']).reset_index()

    if df_airmeasure is None:
        df_airmeasure = df.loc[:, airmeasure_columns].drop_duplicates().reset_index(drop=True)
        # insert query
        for i, x in df_airmeasure.iterrows():
            cur.execute(sql_airmeasure, [i+1] + [x[c] for c in airmeasure_columns])
    else:
        df_temp = df.loc[:, airmeasure_columns].drop_duplicates().reset_index(drop=True)
        if (df_temp != df_airmeasure).sum().sum():
            raise ValueError("not equal table")
    for m in df_airmeasure['measurecode'].values:
        df_airquality = df.loc[df['measurecode'] == m, airquality_columns]
        df_airquality['date'] = pd.to_datetime(df_airquality['date']-1, format='%Y%m%d%H').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # insert row
        for i, x in df_airquality.iterrows():
            idx += 1
            cur.execute(sql_airquality, [idx] + [x[c] for c in airquality_columns])
cur.close()


# # Holiday Data

# In[8]:


from argparse import ArgumentParser
from bs4 import BeautifulSoup as Soup
import requests
from datetime import datetime as dt

parser = ArgumentParser()
parser.add_argument("-i", "--interval", default=1, type=int)
parser.add_argument("-s", "--start", default=2018, type=int)
parser.add_argument("-e", "--end", default=2021, type=int)
parser.add_argument("-c", "--country", default="south-korea")
args = parser.parse_known_args()[0]


# In[9]:


def get_holiday_data(html):
    tables = html.find('table', attrs={"id": "holidays-table"}).find("tbody")
    rows = tables.find_all('tr')

    data = [("date", "day", "name", "type")]
    for r in rows:
        tags = r.find_all("td")
        if len(tags) == 0:
            continue
        date = r.attrs["data-date"]
        date = dt.fromtimestamp(int(int(date) / 1e3))
        new_row = [f"{date.year:04d}-{date.month:02d}-{date.day:02d}"] + [x.text.strip() for x in tags]
        data.append(new_row)
    return data

def craw_data(year, country="south-korea"):
    url = f"https://www.timeanddate.com/holidays/{country}/{year}"
    r = requests.get(url)
    html = Soup(r.text, "html5lib")
    return html


# In[10]:


all_data = []
for y in range(args.start, args.end+1):
    html = craw_data(y, args.country)
    data = get_holiday_data(html)
    all_data.append(pd.DataFrame(data[1:], columns=data[0]))
    
df_holiday = pd.concat(all_data)


# In[12]:


df_holiday = df_holiday.reset_index(drop=True)
df_holiday.to_csv(data_path / f"holiday_{args.start}-{args.end}.tsv", sep="\t", index=False)


# # Weather Data
# 
# ASOS: 종관기상관측이란 종관규모의 날씨를 파악하기 위하여 정해진 시각에 모든 관측소에서 같은 시각에 실시하는 지상관측을 말합니다.
# 
# 종관규모는 일기도에 표현되어 있는 보통의 고기압이나 저기압의 공간적 크기 및 수명을 말하며, 주로 매일의 날씨 현상을 뜻합니다.
# 
# https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36
# 
# - 1회 조회 가능 최대 기간: 분 1일, 시간 1년, 일 10년, 월·연 제한 없음(장기간 자료는 '파일셋 조회' 메뉴 이용)
# - 시간/분 자료에 대해 관측값의 정상 여부를 판단하는 품질검사 플래그(QC FLAG) 정보 제공
#     * 제공 요소: 기온, 습도, 기압, 지면온도, 풍향, 풍속, 일조 / 플래그 종류(의미): 0(정상), 1(오류), 9(결측)
# 
# - 전일 자료는 당일 10시 이후 확인 가능
# 
# http://web.kma.go.kr/communication/knowledge/spring_list.jsp?bid=spring&mode=view&num=247
# 
# mid cloud?
# 
# https://www.scienceall.com/%EC%A4%91%EC%B8%B5-%EA%B5%AC%EB%A6%84middle-cloud/
# 
# 현상번호?
# 
# https://data.kma.go.kr/community/board/selectBoardList.do?bbrdTypeNo=3&pgmNo=95
# 
# 시정?
# 
# https://ko.wikipedia.org/wiki/%EC%8B%9C%EC%A0%95

# In[15]:


columns = {
    '지점': 'measurecode', 
    '지점명': 'measurename', 
    '일시': 'date', 
    '기온(°C)': 'temperature', 
    '강수량(mm)': 'precipitation', 
    '풍속(m/s)': 'windspeed', 
    '풍향(16방위)': 'winddirection', 
    '습도(%)': 'humidity', 
    '현지기압(hPa)': 'spotatmosphericpressure', 
    '지면온도(°C)': 'groundtemperature'
}
{
    '지점': 'measure_code', 
    '지점명': 'measure_name', 
    '일시': 'date', 
    '기온(°C)': 'temperature', 
    '기온 QC플래그': 'temperature-flag',
    '강수량(mm)': 'precipitation', 
    '강수량 QC플래그': 'precipitation-flag',
    '풍속(m/s)': 'wind_speed', 
    '풍속 QC플래그': 'wind_speed-flag',
    '풍향(16방위)': 'wind_direction', 
    '풍향 QC플래그': 'wind_direction-flag', 
    '습도(%)': 'humidity', 
    '습도 QC플래그': 'humidity-flag',
    '증기압(hPa)': 'vapor_pressure', 
    '이슬점온도(°C)': 'dew_point_temperature', 
    '현지기압(hPa)': 'local_pressure', 
    '현지기압 QC플래그': 'local_pressure-flag', 
    '해면기압(hPa)': 'sea_​​level_pressure',
    '해면기압 QC플래그': 'sea_​​level_pressure-flag', 
    '일조(hr)': 'sunshine', 
    '일조 QC플래그': 'sunshine-flag', 
    '일사(MJ/m2)': 'solar_radiation', 
    '일사 QC플래그': 'solar_radiation-flag', 
    '적설(cm)': 'snow',
    '3시간신적설(cm)': 'snow_3hour', 
    '전운량(10분위)': 'cloud', 
    '중하층운량(10분위)': 'mid_level_cloud',
    '운형(운형약어)': 'cloud_type', 
    '최저운고(100m )',
    '시정(10m)': 'visibility', 
    '지면상태(지면상태코드)': 'ground_status_code', 
    '현상번호(국내식)': 'weather_status_code', 
    '지면온도(°C)': 'ground_temperature', 
    '지면온도 QC플래그': 'ground_temperature-flag',
    '5cm 지중온도(°C)': '5cm_soil_temperature', 
    '10cm 지중온도(°C)': '10cm_soil_temperature', 
    '20cm 지중온도(°C)': '20cm_soil_temperature', 
    '30cm 지중온도(°C)': '30cm_soil_temperature'
}

data = []
for year in [2018, 2019, 2020, 2021]:
    df = pd.read_csv(data_path / "weather" / f"{year}년.csv", encoding="euc-kr")
    # df = df.rename(columns=columns).iloc[:, 2:]
    data.append(df)
df = pd.concat(data).reset_index(drop=True)


# In[28]:


len(df['시정(10m)'].unique())


# In[17]:


df.head()


# In[ ]:




