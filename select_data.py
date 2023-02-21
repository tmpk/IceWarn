# script for perusal of radiosonde/ceilometer data,
# and performing a selection of appropriate datasets

from scipy.interpolate import interp1d
import numpy as np
from netCDF4 import Dataset 
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import pandas as pd
import datetime
from scipy.signal import argrelextrema
import requests
from utils import date_to_string, get_period, leap_year
import glob

## create a custom colormap ranging from light grey through 
# blue, green, yellow and dark red
cdict = {'red':   ((0.0,   0.8, 0.8),
                   (0.25,  0.0, 0.0),
                   (0.5,   0.0, 0.0),
                   (0.625, 1.0, 1.0),
                   (0.875, 1.0, 1.0),
                   (1.0,   0.65, 0.65)),
                   

         'green': ((0.0,   0.8, 0.8),
                   (0.25,  0.0, 0.0),
                   (0.5,   1.0, 1.0),
                   (0.625, 1.0, 1.0),
                   (0.75,  0.0, 0.0),
                   (1.0,   0.0, 0.0)),

         'blue':  ((0.0,   0.8, 0.8),
                   (0.25,  1.0, 1.0),
                   (0.5,   0.0, 0.0),
                   (0.625, 0.0, 0.0),
                   (0.75,  0.0, 0.0),
                   (1.0,   0.0, 0.0))
        }
custom_cmap = matplotlib.colors.LinearSegmentedColormap('Custom_Cmap', cdict)
plt.register_cmap(cmap=custom_cmap)

# filenames = []
# filepaths = glob.glob("data\selected_data\\andoya\*")
# for path in filepaths:
#     filename = str.rsplit(path, '\\')[-1]
#     filename = str.rsplit(filename, '.')[0]
#     filenames.append(filename)


def run(y, m, d, location): 
    #pd.set_option('display.expand_frame_repr', False)
    period = get_period(y, m)
    y, m, d = date_to_string(y, m, d)
    ds = xr.open_dataset(f'data/Radiosonde/andoya/{y}/andoya_' + period + '.nc')
    
    data_rs = ds.to_dataframe()
    
    sample_time = [tuple[0]+tuple[1] for tuple in data_rs.index]
    start_time_rs = data_rs.index.get_level_values(0)
    sid = start_time_rs
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

    # load data from ceilometer:
    data_cm = Dataset(f'data/Ceilometer/andoya/{y}/{m}/{d}/dataset.nc')
    
    # loop over radiosonde timeseries for the given day:
    for sid in unique_sids:
        ### RADIOSONDE DATA
        plot_data = grouped_rs_data.get_group(sid)
        
        time = pd.to_datetime(plot_data['UTC time'], utc=True)
        # time = np.array(time.dt.to_pydatetime())        # radiosonde launch time
        starttime_rs = time[0]
        stoptime_rs = time[-1]
        starttime_rs_string = starttime_rs.strftime('%Y-%m-%d %H:%M:%S')
        stoptime_rs_string = stoptime_rs.strftime('%Y-%m-%d %H:%M:%S')
        pd.set_option('display.max_columns', 100)

        #if not starttime_rs_string.rsplit(" ")[-1] == stime:
        #    continue
        
        rhs = plot_data['relative_humidity']
        alts = plot_data['altitude']
        temps = plot_data['air_temperature'] - 273.15   
        pressure = plot_data['air_pressure']    
        ####
        ### CEILOMETER DATA
        start_time_cm = data_cm.variables['start_time'][:]
        altitude = data_cm.variables['altitude'][:]
        cloud_base_heights = data_cm.variables['cloud_base_height'][:]
        
        att_bsc = data_cm.variables['attenuated_backscatter_0'][:] * 10**(-6)
        start_time_cm_timestamp = np.array([(t * 60*60*24) for t in start_time_cm])
        # convert timestamps to utc datetime
        start_time_cm_datetime = np.array([datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t in start_time_cm_timestamp]) #np.array([datetime.datetime.utcfromtimestamp(t) for t in start_time_cm_timestamp])
        start_time_cm_datetime = pd.to_datetime(start_time_cm_timestamp, unit='s', utc=True)
        
        indices = np.nonzero( (start_time_cm_datetime > time[0]-datetime.timedelta(minutes=10)) & (start_time_cm_datetime < time[-1]-datetime.timedelta(minutes=5)) )[0]
        if (indices[-1] == len(start_time_cm_datetime)-1):
            indices = indices[:-1]
        ids = indices[1:3]

        n = (time[0] - start_time_cm_datetime[indices[1]]).total_seconds()
        avg_used = False
        first = False
        if n > 90 and n < 210:
            alpha = (300-n)/300
            beta = 1-alpha
            cm_avg = np.add(alpha*att_bsc[:,ids[0]], beta*att_bsc[:,ids[1]])
            if np.abs(cloud_base_heights[ids[0],0] - cloud_base_heights[ids[1], 0]) > 1000:
                cbh = np.min([ cloud_base_heights[ids[0],0], cloud_base_heights[ids[1], 0] ])
            else:
                cbh = np.add(alpha*cloud_base_heights[ids[0],0], beta*cloud_base_heights[ids[1], 0])
            cm = cm_avg
            avg_used = True
        elif n < 90:
            cm = att_bsc[:, ids[0]]
            cbh = cloud_base_heights[ids[0],0]
            first = True
        else:
            cm = att_bsc[:, ids[1]]
            cbh = cloud_base_heights[ids[1],0]

        ### CHECKS
        index = np.argmax(rhs)  # index of maximum humidity for radiosonde timeseries in question
        if alts[index] > 2000:  # skip sample if rh max is above 2000 m
            print("RH max is above 2000m. Continuing with next...")
            continue

       
        if cbh is np.ma.masked or np.isnan(cbh):
            print("No cloud base height detected; indicating clear air. Continuing with next..")
            continue            # no cloud base height means clear air; skip
        ###

        ### ASSIGN MAX HEIGHT FOR WHICH TO CONDUCT MAPPING
        extrema = argrelextrema(cm, comparator=np.greater, order=10)
        for extremum in extrema[0]:
            if altitude[extremum]+100 >= cbh:
                max_h = altitude[extremum]
                break
        
        f = interp1d(alts, rhs)

        altitude_um = np.ma.compressed(altitude)
        idxs = altitude_um <= max_h
        idxs2 = alts <= max_h
        altitude_um = altitude_um[idxs]
        rh_clm = f(altitude_um)
        
        plt.rcParams['axes.grid'] = True
        #--------------------------------
        ################# plot cm measurements
        ctr=0
        fig3, ax3 = plt.subplots()
        mat = np.empty((len(altitude), len(indices)))
        for i in indices:
            for h in cloud_base_heights[i]:
                # plot a horizontal line representing cloud base layer for each 5 min period
                x1 = start_time_cm_datetime[i]
                y1 = h
                x2 = start_time_cm_datetime[i+1] 
                y2 = y1
                ax3.plot([x1, x2], [y1, y2], color='w', linewidth=1.5)
            mat[:, ctr] =  att_bsc[:, i] # np.log10(att_bsc[:, i])
            ctr += 1
        mat = np.ma.masked_where(mat==1.0,mat)
        plot3 = ax3.imshow(mat, extent=[matplotlib.dates.date2num(start_time_cm_datetime[indices[0]]), matplotlib.dates.date2num(start_time_cm_datetime[indices[-1]+1]), 0, altitude[-1]], 
                            aspect='auto', origin='lower', cmap='Custom_Cmap', interpolation='none') #, vmin=-2.0, vmax=2.0)
        ax3.legend([r'Cloud base'], loc=4, facecolor='k',labelcolor='white')
        ax3.xaxis.set_tick_params(rotation=45)
        cbar3 = plt.colorbar(plot3, ax=ax3)
        cbar3.set_label(r'Attenuated backscatter')
        ax3.grid(b=True, which='both')
        ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax3.set_xlabel(r'Time [UTC]')
        ax3.set_ylabel(r'Altitude [m]')
        ax3.set_ylim(0, max_h+500)
        
        ##################
        fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
        ax[0].plot(att_bsc[:, indices[0]], altitude)
        ax[0].set_title(f"Ceilometer reading, {start_time_cm_datetime[indices[0]]}")
        ax[1].plot(att_bsc[:, indices[1]], altitude)
        ax[1].set_title(f"Ceilometer reading, {start_time_cm_datetime[indices[1]]}")
        ax[2].plot(att_bsc[:, indices[2]], altitude)
        ax[2].set_title(f"Ceilometer reading, {start_time_cm_datetime[indices[2]]}")
        ax[3].plot(rhs, alts)
        ax[3].set_title(f"Radiosonde launched {starttime_rs_string}")
        ax[0].set_ylim(0, max_h+500)
        ax[3].set_xlim(np.min(rhs[idxs2])-2, 102)
        ax[0].ticklabel_format(axis='x', style='sci', scilimits=(-5,-5))
        ax[1].ticklabel_format(axis='x', style='sci', scilimits=(-5,-5))
        ax[2].ticklabel_format(axis='x', style='sci', scilimits=(-5,-5))
        ax[0].get_shared_x_axes().join(ax[0], *ax[1:3])
        # --------------------------------
        if avg_used:
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)

        for a, i in zip(ax[0:2], ids):
            a.plot(att_bsc[:, i], altitude)
            a.hlines(cloud_base_heights[i], 0, np.max(cm), colors='0.75')
            a.set_title(f"Ceilometer reading, {start_time_cm_datetime[i]}")
            a.set_xlabel(r'Attenuated backscatter [m$^{-1}$ sr$^{-1}$]')
        ax[0].set_ylabel('Altitude')
        ax[0].set_ylim(0, max_h+500)
        if avg_used:
            ax[2].plot(cm, altitude)
            ax[2].set_title('Time-weighted avg. of clm readings')
            ax[2].hlines(max_h, 0, np.max(cm), colors='k')
        if first:
            ax[0].hlines(max_h, 0, np.max(cm), colors='k')
        else:
            ax[-2].hlines(max_h, 0, np.max(cm), colors='k')
        ax[-1].plot(rhs, alts)
        ax[-1].set_title(f"Radiosonde launched {starttime_rs_string}")
        ax[-1].set_xlim(np.min(rhs[idxs2])-2, 102)
        ax[-1].set_xlabel('RH [%]')
        # -----------------------------------
        fig, axx = plt.subplots(nrows=1, ncols=4)
        pot_temp = (temps[idxs2] + 273.15)*(1000 / pressure[idxs2])**0.286 - 273.15 
        axx[0].plot(temps[idxs2], alts[idxs2])
        axx[0].plot(pot_temp, alts[idxs2])
        axx[0].legend(['air temperature', 'potential temperature'])
        axx[0].set_xlabel('Temperature')
        axx[0].set_ylabel('Altitude')
        axx[1].plot(rhs[idxs2], alts[idxs2])
        axx[1].set_xlabel('RH')
        axx[1].set_xlim(np.min(rhs[idxs2])-2, 102)
        axx[2].plot(cm[idxs], altitude_um)
        axx[2].set_xlabel('Backscatter')
        axx[0].sharey(axx[1])
        axx[1].sharey(axx[2])
        scplt = axx[3].scatter(cm[idxs], rh_clm, c=altitude_um, cmap='gist_rainbow_r')
        plt.colorbar(scplt, ax=axx[3])
        axx[3].set_xlabel('Backscatter')
        axx[3].set_ylabel('RH')
        axx[3].set_ylim(np.min(rhs[idxs2])-2, 102)
        
        axx[3].ticklabel_format(axis='x', style='sci', scilimits=(-5,-5))
        axx[2].ticklabel_format(axis='x', style='sci', scilimits=(-5,-5))
        #---------------------------
        temp_avgd = np.convolve(temps[idxs2], np.ones((3,))/3, mode='valid')
        rhs_avgd = np.convolve(rhs[idxs2], np.ones((3,))/3, mode='valid')
        pot_temp_diffs = np.diff(pot_temp)     #  temp_avgd) # 
        rhs_diffs = np.diff(rhs[idxs2])         # rhs_avgd)   # 
        alts_v = alts[idxs2]
        alt_diffs = np.diff(alts[idxs2])        # alts_v[2:]) # 
        
        ### CORRELATION
        temps_deriv = pot_temp_diffs/alt_diffs
        rhs_deriv = rhs_diffs/alt_diffs
        r = np.corrcoef(temps_deriv, rhs_deriv)
        #print("corr: ", r)
        # -----------------
        # f, a = plt.subplots()
        # a.scatter(temps_deriv, rhs_deriv)
        # a.spines['top'].set_color('none')
        # a.spines['left'].set_position('zero')
        # a.spines['right'].set_color('none')
        # a.spines['bottom'].set_position('zero')
        
        
        # a.annotate('dT/dh', xy=(1, 0), xycoords=('axes fraction', 'data'), 
        # xytext=(10, -10), textcoords='offset points',
        # ha='center', va='center',
        # )
        # a.annotate('d(RH)/dh', xy=(0, 1), xycoords=('data', 'axes fraction'), 
        # xytext=(0, 10), textcoords='offset points',
        # ha='center', va='center',
        # )
        # yabs_max = abs(max(a.get_ylim(), key=abs))
        # a.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        # xabs_max = abs(max(a.get_xlim(), key=abs))
        # a.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        # a.grid(False)
        #-------------------------
        max_h_old = max_h
        rhsmax_idx = np.argmax(rhs)
        rhs2 = rhs[1 + rhsmax_idx::]
        alts2 = alts[1 + rhsmax_idx::]
        rhsmax_idx2 = np.argmax(rhs2)
     
        if alts[rhsmax_idx] > max_h_old:
            max_h = alts[rhsmax_idx]
            idxs2 = (alts.le(max_h, fill_value=np.inf)).values
        elif alts2[rhsmax_idx2] > max_h_old:
            max_h = alts2[rhsmax_idx2]
            idxs2 = (alts.le(max_h, fill_value=np.inf)).values
        else:
            tmp = alts[alts > max_h_old]
            max_h = tmp[0]
            idxs2 = (alts.le(max_h, fill_value=np.inf)).values
        
        #delta = np.abs(att_bsc[idxs, ids[1]] -  att_bsc[idxs, ids[0]])
        #dvs = np.maximum(att_bsc[idxs, ids[1]], att_bsc[idxs, ids[0]])
        #print(delta)
        #print(dvs)
        #print(np.divide(delta, dvs))

        plt.show()
      
     
        str = input("store? y/n: ")
        filename = starttime_rs_string.split()[0] + '_' + starttime_rs_string.split()[1].replace(':', '-')
        
        if str == 'y': #if filename in filenames:
            data1 = {'alt': alts[idxs2].values, 'rh': rhs[idxs2].values, 'temp': temps[idxs2].values, 'pressure': pressure[idxs2].values}
            data2 = {'alt': altitude_um, 'att_bsc': cm[idxs]}
            df1 = pd.DataFrame.from_dict(data1)
            df1['att_bsc'] = '-'
            df2 = pd.DataFrame.from_dict(data2)
            df2['rh'] = '-'
            df2['temp'] = '-'
            df2['pressure'] = '-'
            df = pd.concat([df1, df2])
            df.sort_values(by=['alt'], inplace=True, ignore_index=True)
            #filename = starttime_rs_string.split()[0] + '_' + starttime_rs_string.split()[1].replace(':', '-')
            df.to_csv(f'data\selected_data\\andoya\{filename}.csv')
            print("saved file: ", filename)
        if str == 'n':
            continue

run(2019, 11, 17, 'andoya')
# fpaths = glob.glob(f"data\selected_data\\andoya/*")

# for fpath in fpaths:
#     s = fpath.rsplit('\\')[-1]
#     s = s.rsplit('.')[0]
#     s = s.rsplit("_")
#     date = s[0] 
#     date_split = date.split('-')
#     y = int(str.strip(date_split[0]))
#     m = int(str.strip(date_split[1]))
#     d = int(str.strip(date_split[2])) 
#     time = s[1].replace("-", ":")
    
#     run(y, m, d, time, 'andoya')

