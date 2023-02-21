import os
from netCDF4 import Dataset
import requests
import datetime

def leap_year(year):
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                is_leap = True
            else:
                is_leap = False
        else:
            is_leap = True
    else:
        is_leap = False
    return is_leap

def get_period(year, month):
    # data from MET's thredds server offers data for one whole month. 
    # This function returns the filename format of the one-month period 
    # we are interested in  
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
          '12']
    previous_month = months[month - 2]
    month = months[month - 1]
    # -2 not -1 becuase index from 0 and the data is saved from the last day
    # of the previous month in MET database
    
    thirty = ['04', '06', '09', '11']
    thirty_one = ['01', '03', '05', '07', '08', '10', '12']
    
    if previous_month in thirty_one:
        if previous_month == '12':
            previous_year = year - 1
            period_1 = f'{previous_year}1231'
        else:
            period_1 = f'{year}{previous_month}31'
    elif previous_month in thirty:
        period_1 = f'{year}{previous_month}30'
    elif previous_month == '02':
        if leap_year(year):
            period_1 = f'{year}{previous_month}29'
        else:
            period_1 = f'{year}{previous_month}28'
    
    if month in thirty_one:
        period_2 = f'-{year}{month}31'
    elif month in thirty:
        period_2 = f'-{year}{month}30'
    elif month == '02':
        if leap_year(year):
            period_2 = f'-{year}{month}29'
        else:
            period_2 = f'-{year}{month}28'
    
    period = period_1 + period_2
    return period

def make_directory(year):
    y = str(year)
    directory = f'./data/Radiosonde/andoya/{y}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def download_month(year, month, directory):
    period = get_period(year, month)
    year = str(year)
    month = '0' + str(month) if month < 10 else str(month)
    core ="https://thredds.met.no/thredds/fileServer/remotesensingradiosonde"
    file_url = core + f"/{year}/{month}/andoya_" + period + ".nc"
    file_name = directory + file_url.rsplit('/', 1)[-1]
    print("filename: ", file_name)
    print("downloading: ", file_url)
    r = requests.get(file_url, stream=True)
    with open(file_name, "wb") as nc:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                nc.write(chunk)

def download(year, month):
    directory = make_directory(year)
    if month=='all':
        now = datetime.datetime.now()
        current_year = now.strftime("%Y")
        current_month = now.strftime("%m")
        if year == int(current_year):
            months = range(1, int(current_month)+1)
        else:
            months = range(1,13)
        for m in months:
            download_month(year, m, directory)
    else:
        download_month(year, month, directory)

for m in range(2, 3):
    download(2019, m)