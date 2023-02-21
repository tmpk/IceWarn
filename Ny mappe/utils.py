import requests 
import pandas as pd

def date_to_string(year, month, day):
    year = str(year)
    month = '0' + str(month) if month < 10 else str(month)
    day = '0' + str(day) if day < 10 else str(day)
    return year, month, day

def get_period(year, month):
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

def leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def get_MET_data(station, data, period):
    client_id = '3695f017-f71e-43bc-b3e0-8a0fb90c2608'
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
            'sources': station,                           
            'elements': data, 
            'referencetime': period,
    }

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    print("Issued GET request to frost API...")
    # Extract JSON data
    json = r.json()

    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])

    # This will return a Dataframe with all of the observations in a table format
    df = pd.DataFrame()
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        df = df.append(row)

    # Retain the following columns
    columns = ['sourceId','referenceTime','elementId','value','unit','timeOffset', 'timeResolution']
    df2 = df[columns].copy()
    # Convert the time value to datetime
    df2['referenceTime'] = pd.to_datetime(df2['referenceTime'])

    return df2
