###
# retrieves precipitation data from MET stations for a set of dates and times
# and stores it in .xlsx format
###

import numpy as np
from netCDF4 import Dataset 
import pandas as pd
import datetime
import requests
import xarray as xr
from utils import date_to_string, get_period

def get_MET_data(station, data, period):
    # retrieves desired "data" during "period" from a given MET "station"
    # by accessing MET's Frost API
     
    client_id = '3695f017-f71e-43bc-b3e0-8a0fb90c2608'
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
            'sources': station,                           
            'elements': data, 
            'referencetime': period,
    }

    # Issue GET request
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
    df2['referenceTime'] = pd.to_datetime(df2['referenceTime'],utc=True)

    return df2

def run(y, m, d): 
    pd.set_option('display.expand_frame_repr', False)
    period = get_period(y, m)
    y, m, d = date_to_string(y, m, d)
    ds = xr.open_dataset(f'data/Radiosonde/andoya/{y}/andoya_' + period + '.nc')
    data_rs = ds.to_dataframe()
    sample_time = [tuple[0]+tuple[1] for tuple in data_rs.index]
    start_time_cm = data_rs.index.get_level_values(0)
    sid = start_time_cm
    data_rs.insert(1, 'sampleseries_id', sid)
    data_rs.insert(0, "UTC time", sample_time)

    # get the radiosonde data for the day of interest:
    dates = pd.to_datetime(data_rs['sampleseries_id']).dt.date
    desired_date = datetime.date(year=int(y), month=int(m), day=int(d))
    idx = (dates == desired_date)
    
    if len(idx[idx == True]) == 0:
        print(f"Error: Unavailable radiosonde data for {d}.{m}.{y}")
        return

    data = data_rs.loc[idx]
    unique_sids = data['sampleseries_id'].unique()
    grouped_rs_data = data.groupby('sampleseries_id') 
    next_date = desired_date + datetime.timedelta(days=1)
    precip_data = get_MET_data(station='SN87110', data='sum(precipitation_amount PT10M)', period=f'{desired_date}/{next_date}')
    
    out = []
    for sid in unique_sids:
        ### RADIOSONDE DATA
        plot_data = grouped_rs_data.get_group(sid)
        time = pd.to_datetime(plot_data['UTC time'], utc=True)
        # time = np.array(time.dt.to_pydatetime())        # radiosonde launch time
        starttime_rs = time[0]
        stoptime_rs = time[-1]
        starttime_rs_string = starttime_rs.strftime('%Y-%m-%d %H:%M:%S')
        stoptime_rs_string = stoptime_rs.strftime('%Y-%m-%d %H:%M:%S')
        
        ### precipitation
        p0 = starttime_rs - datetime.timedelta(minutes=60)
        p1 = starttime_rs + datetime.timedelta(minutes=60)
        precipitation = precip_data[precip_data['referenceTime'].between(p0, p1)]  # precipitation around radiosonde launch time
        
        if (precipitation.value > 0).any() :
            col = precipitation['referenceTime'].dt.tz_localize(None)
            precipitation = precipitation.assign(referenceTime=col)
            if len(out)==0:
                out = precipitation
            else:
                out = pd.concat([out, precipitation])

    if len(out)==0:
        return 0 
    else:
        return out

with open('dates_measurements.txt') as f:
    next(f)
    i = 0
    for line in f:
        if not line.strip():
            break
        date = line.rsplit(' ')[0]
        split = date.split('.')
        d = int(str.strip(split[0]))
        m = int(str.strip(split[1]))
        y = int('20' + str.strip(split[2])) 

        df2 = run(y, m, d)
        if type(df2) == int:
            continue
        if i == 0:
            output = df2
        else:
            output = pd.concat([output, df2])
        i+=1
    
    with pd.ExcelWriter('precipitation.xlsx') as writer:
        output.to_excel(writer) 