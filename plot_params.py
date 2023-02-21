# script for loading radiosonde and ceilometer data 
# and creating various plots to investigate the data

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime 
import xarray as xr
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)
from netCDF4 import Dataset 
from scipy.interpolate import interp1d, splrep, splev, splprep
from scipy.signal import butter, freqz, lfilter, argrelextrema
from utils import date_to_string, get_period, leap_year

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
####

# def days_in_month(month, year):
#     lst = ['01', '03', '05', '07', '08', '10', '12']
#     if not in lst:
#         if (month == '02'):
#             if leap_year(int(year)):
#                 return 29
#             else:
#                 return 28
#         else:
#             return 30
#     else:
#         return 31


def run(y, m, d):

    period = get_period(y, m)
    y, m, d = date_to_string(y, m, d)

    ds = xr.open_dataset(f'data/Radiosonde/andoya/{y}/andoya_' + period + '.nc')
    
    data_rs = ds.to_dataframe()
    sample_time = [tuple[0]+tuple[1] for tuple in data_rs.index]
    start_time = data_rs.index.get_level_values(0)
    sid = start_time
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
    #data['UTC time'] = pd.to_datetime(data['UTC time'])

    # (usually) two radiosonde timeseries for each day; group them by ID:
    unique_sids = data['sampleseries_id'].unique()
    grouped_rs_data = data.groupby('sampleseries_id') 

    # load data from ceilometer:
    data_cm = Dataset(f'data/Ceilometer/andoya/{y}/{m}/{d}/dataset.nc')
 
    # loop over radiosonde timeseries:
    for sid in unique_sids:
        fig = plt.figure(1)
        #ax1 = plt.subplot(321) # scatter rh
        #ax2 = plt.subplot(323,sharex=ax1,sharey=ax1) # scatter dp
        ax4 = plt.subplot2grid((2,3), (0,0)) #plt.subplot(222, sharey=ax3) # rh
        ax7 = plt.subplot2grid((2,3), (0,1), sharey=ax4)
        ax3 = plt.subplot2grid((2,3), (1,0), sharey=ax4,colspan=2) # plt.subplot(221) # bsc
        ax5 = plt.subplot2grid((2,3), (0,2), sharey=ax4) #plt.subplot(223,sharey=ax3) # dp depr
        ax6 = plt.subplot2grid((2,3), (1,2), sharey=ax4) # plt.subplot(224,sharey=ax3) # temp, dp
        plot_data = grouped_rs_data.get_group(sid)
        time = pd.to_datetime(plot_data['UTC time'])
        time = np.array(time.dt.to_pydatetime())
        starttime = time[0]
        stoptime = time[-1]
        starttime_string = starttime.strftime('%Y-%m-%d %H:%M:%S')
        
        #if not str.rsplit(starttime_string, " ")[-1] == stime:
        #    continue

        stoptime_string = stoptime.strftime('%Y-%m-%d %H:%M:%S')
        pd.set_option('display.max_columns', 100)
        rhs = plot_data['relative_humidity']
        alts = plot_data['altitude']
        temps = plot_data['air_temperature'] - 273.15           # temperatures in Celsius
        dp_temps = plot_data['dew_point_temperature'] - 273.15  # temperatures in Celsius
        spread = temps-dp_temps

        #plot1 = ax1.scatter(time, alts, c=rhs, cmap='gist_rainbow_r', marker='_', linewidth=2.5) 
        #cbar1 = plt.colorbar(plot1, ax=ax1)
        #cbar1.set_label('RH [%]')
        #ax1.set_ylabel(r'Altitude [m]')
        #ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        #ax1.grid(b=True, which='both')
        
        # calculate air density
        R_s = 287.058 # specific gas constant, dry air. Units J/(kg*K)
        pressure = plot_data['air_pressure']*100 # plot_data['air_pressure'] is in hPa -> convert to Pa
        rho = pressure/(R_s*(temps + 273.15))
        ##

        # plot RH, temperature against altitude
        #plot2 = ax2.scatter(time, alts, c=temps-dp_temps, cmap='gist_rainbow', marker='_', linewidth=2.5)
        #cbar2 = plt.colorbar(plot2, ax=ax2)
        #cbar2.set_label(r'Dewpoint depression [$^{\circ}$C]')
        #ax2.set_ylabel('Altitude [m]')
        #ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        #ax2.grid(b=True, which='both')
        ##
        
        # plot backscatter and cloud base height from ceilometer
        start_time = data_cm.variables['start_time'][:]
        altitude = data_cm.variables['altitude'][:]
        cloud_base_heights = data_cm.variables['cloud_base_height'][:]
        att_bsc = data_cm.variables['attenuated_backscatter_0'][:]
        start_time_timestamp = np.array([(t * 60*60*24) for t in start_time])
        start_time_datetime = np.array([datetime.datetime.utcfromtimestamp(t) for t in start_time_timestamp])
        indices = np.nonzero( (start_time_datetime > time[0]-datetime.timedelta(minutes=10)) & (start_time_datetime < time[-1]-datetime.timedelta(minutes=5)) )[0]
        
        ctr = 0
        if (indices[-1] == len(start_time_datetime)-1):
            indices = indices[:-1]
        mat = np.empty((len(altitude), len(indices)))
        for i in indices:
            for h in cloud_base_heights[i]:
                # plot a horizontal line representing cloud base layer for each 5 min period
                x1 = start_time_datetime[i]
                y1 = h
                x2 = start_time_datetime[i+1] 
                y2 = y1
                ax3.plot([x1, x2], [y1, y2], color='w', linewidth=1.5)
            mat[:, ctr] =  att_bsc[:, i] # np.log10(att_bsc[:, i])
            ctr += 1
        mat = np.ma.masked_where(mat==1.0,mat)
        plot3 = ax3.imshow(mat, extent=[matplotlib.dates.date2num(start_time_datetime[indices[0]]), matplotlib.dates.date2num(start_time_datetime[indices[-1]+1]), 0, altitude[-1]], 
                            aspect='auto', origin='lower', cmap='Custom_Cmap', interpolation='none') #, vmin=-2.0, vmax=2.0)
        ax3.legend([r'Cloud base'], loc=4, facecolor='k',labelcolor='white')
        ax3.xaxis.set_tick_params(rotation=45)
        cbar3 = plt.colorbar(plot3, ax=ax3)
        cbar3.set_label(r'Attenuated backscatter')
        ax3.grid(b=True, which='both')
        ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax3.set_xlabel(r'Time [UTC]')
        ax3.set_ylabel(r'Altitude [m]')
        cbh_max = np.ceil(np.amax(cloud_base_heights[indices, :]))
        
        if type(cbh_max) != np.dtype(np.float64) or np.isnan(cbh_max):
            cbh_max = 1500
        if cbh_max > 2000:
            yticks = np.arange(0,cbh_max+200, 400)
        else:
            yticks = np.arange(0,cbh_max+200, 200)
        
        ax3.set_ylim(0, cbh_max+200)
        ax3.set_yticks(yticks)
        ind = np.where(alts > cbh_max+200)[0]
        ax3.set_xlim(starttime, time[ind[0]])
        ax3.yaxis.set_minor_locator(AutoMinorLocator(2))

        #fig.suptitle(f'Radiosonde measurements for {starttime_string} - {stoptime_string}')
        #plt.show()
        
        #fig, (ax4, ax5) = plt.subplots(2,1)
        dp_depression = temps-dp_temps
        plot_data_h = plot_data.loc[plot_data['geopotential_height'] < cbh_max+200]
        if plot_data_h.empty == False:
            dp_depr = plot_data_h['air_temperature'] - plot_data_h['dew_point_temperature'] 
            air_temp = plot_data_h['air_temperature'] - 273.15
            hmds = plot_data_h['relative_humidity']
        else:
            dp_depr = temps - dp_temps
            air_temp = temps
            hmds = rhs
        ax4.plot(rhs, alts)
        ax4.grid(b=True, which='both')
        ax4.set_ylabel(r'Altitude [m]')
        ax4.set_xlabel(r'RH [%]')
        ax4.set_ylim(0, cbh_max+200)
        ax4.set_xlim(np.min(hmds)-5, np.max(hmds)+5)
        ax4.set_yticks(yticks)
        ax4.yaxis.set_minor_locator(AutoMinorLocator(2))

        
        # ax5.plot(dp_depression, alts, )
        # ax5.grid(b=True, which='both')
        # ax5.set_ylabel(r'Altitude [m]')
        # ax5.set_xlabel(r'Dewpoint depression [$^{\circ}$C]')
        # ax5.set_xlim(np.nanmin(dp_depr)-1, np.nanmax(dp_depr)+1)


        ax6.plot(temps, alts)
        ax6.plot(dp_temps, alts)
        ax6.legend(['Ambient temperature', 'Dewpoint temperature'])
        ax6.set_ylabel(r'Altitude [m]')
        ax6.set_xlabel(r'Temperature [$^{\circ}$C]')
        ax6.set_xlim(np.nanmin(air_temp)-1, np.nanmax(air_temp)+1)
        ax6.grid(b=True, which='both')
        ax4.set_title(f"Radiosonde measurement starting at {starttime_string}")

        fig1, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
        ax[0].set_ylim(0, cbh_max+200)
        #ax[0].set_yticks(yticks)
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0].plot(rhs, alts)
        ax[0].set_xlim(np.min(hmds)-5, np.max(hmds)+5)
        ax[0].grid(b=True, which='both')
        ax[0].set_ylabel(r'Altitude [m]')
        ax[0].set_xlabel(r'RH [%]')
        ax[0].set_title(f"Radiosonde launch at {starttime_string}")
        ids = indices[1:3]
        for a, i in zip(ax[1::], ids):
            a.plot(att_bsc[:, i], altitude)
            a.set_xlabel(r'Attenuated backscatter')
            a.grid(b=True, which='both')
            a.set_title(f"Ceilometer reading, {start_time_datetime[i]}")
            a.grid(b=True, which='both')
            a.set_xlabel(r'Attenuated backscatter [m$^{-1}$ sr$^{-1}$]')
        n = (time[0] - start_time_datetime[indices[1]]).total_seconds()
        alpha = (300-n)/300
        beta = 1-alpha
        cm_avg = np.add(alpha*att_bsc[:,ids[0]], beta*att_bsc[:,ids[1]])
        idx = alts < cbh_max+200
        #rho_interp = interp1d(alts[idx], rho[idx])
        idx2 = altitude < alts[idx].iloc[-1]
        idx2 = np.ma.compressed(idx2)
        altitudes = np.ma.compressed(altitude[idx2]) 
        #rho_m = rho_interp(altitudes) 
        #ax[1].plot(cm_avg[idx2]-rho_m, altitudes) #(np.log10(cm_avg), altitude) #
        #ax[1].plot(cm_avg[idx2], altitudes, "-")
        ax7.plot(cm_avg[idx2], altitudes, "-")
        ax7.grid(b=True, which='both')
        ax7.set_title(f"Ceilometer, avg {start_time_datetime[ids[0]]} & {start_time_datetime[ids[1]]}", fontsize=12)
        ax7.set_xlabel(r'Attenuated backscatter [m$^{-1}$ sr$^{-1}$]')
        #ax[1].plot(rho_m, altitudes, '-.')
        # ax[1].grid(b=True, which='both')
        # ax[1].set_title("Ceilometer")
        # ax[1].set_xlabel(r'Attenuated backscatter [m$^{-1}$ sr$^{-1}$]')

        figg, axxx = plt.subplots()
        axxx.plot()
        axxx.set_ylim(0, cbh_max+200)
        #ax[0].set_yticks(yticks)
        axxx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axxx.plot(rhs, alts,'b')
        axxx.set_xlim(np.min(hmds)-5, np.max(hmds)+5)
        axxx.grid(b=True, which='both')
        axxx.set_ylabel(r'Altitude [m]')
        axxx.set_xlabel(r'RH [%]')
        axxx.xaxis.label.set_color('b')
        axxx.xaxis.label.set_fontsize(20)
        axxx.yaxis.label.set_fontsize(20)
        axxx.tick_params(axis='x', colors='b')
        
        twin1 = axxx.twiny()
        twin1.plot(cm_avg[idx2][:-10], altitudes[:-10], 'r-')
        twin1.set_xlabel(r'Attenuated backscatter [m$^{-1}$ sr$^{-1}$]')
        twin1.xaxis.label.set_color('r')
        twin1.xaxis.label.set_fontsize(20)
        twin1.tick_params(axis='x',colors='r')
        cm_avg_d = []   
        ds = np.count_nonzero(idx2)

        # fit_func = [675*(((0.0015*x)/(102-x))+0.005)-3.5 for x in rhs]
        # fit_func = [5*(1/(100.1-x))+1.5 for x in rhs]
        # twin1.plot(fit_func, alts)

        for i in range(ds-2):
            der = (cm_avg[i+1] - cm_avg[i]) / (altitudes[i+1] - altitudes[i]) 
            cm_avg_d.append( der )

        ax5.plot(cm_avg_d, altitudes[0:ds-2])
        ax5.grid(b=True, which='both')
      
        ax5.set_xlabel(r'd$\beta$/dh')


        ### MAP RH TO ALT
        index = np.argmax(rhs)  # index of maximum humidity for radiosonde timeseries in question

        if alts[index] > 2000:  # skip sample if rh max is above 2000 m
            continue
        
        f = interp1d(alts, rhs)

        altitude_um = np.ma.compressed(altitude)
        cbh = np.add(alpha*cloud_base_heights[ids[0],0], beta*cloud_base_heights[ids[1], 0])
        
        extrema = argrelextrema(cm_avg, comparator=np.greater, order=10)
        first_extrema = extrema[0][0]
        for extrema in extrema[0]:
            if altitude[extrema] > cbh:
                val = altitude[extrema]
                break
            else:
                val = 1500
        
        idxs = altitude_um <= val
        idxs2 = alts <= val
        altitude_um = altitude_um[idxs]
        rh_clm = f(altitude_um)
        print("Mapping done for altitudes <= ", val)
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        ax[0].plot(cm_avg, altitude)
        ax[0].hlines(cbh, 0, np.max(cm_avg), colors='0.75')
        ax[0].set_ylabel('Altitude')
        ax[0].set_xlabel('Backscatter')
        ax[1].plot(rhs, alts)
        ax[0].hlines(val, 0, np.max(cm_avg), colors='k')
        ax[0].get_shared_y_axes().join(ax[0], ax[1])
        ax[0].set_ylim(0, val+200)
        ax[1].set_xlabel('RH')

        fig, axx = plt.subplots(nrows=1, ncols=4)
        axx[0].plot(temps[idxs2], alts[idxs2])
        axx[0].set_xlabel('Temperature')
        axx[0].set_ylabel('Altitude')
        axx[1].plot(cm_avg[idxs], altitude_um)
        axx[1].set_xlabel('Backscatter')
        axx[2].plot(rhs[idxs2], alts[idxs2])
        axx[2].set_xlabel('RH')
        axx[0].sharey(axx[1])
        axx[1].sharey(axx[2])
        axx[3].scatter(cm_avg[idxs], rh_clm)
        axx[3].set_xlabel('Backscatter')
        axx[3].set_ylabel('RH')
        ####
        # fig2, axx = plt.subplots(nrows=1, ncols=2, sharey=True)
        # axx[0].plot(rhs, alts)
        # axx[0].set_xlabel('RH [%]')
        # axx[0].set_ylabel('Altitude [m]')
        # axx[1].plot(spread, alts)
        # axx[1].set_xlabel('Dewpoint depression [C]')
        # axx[1].set_ylabel('Altitude [m]')

        # Magnus formula 
        # b = 18.678
        # c = 257.14
        # gamma = np.log(rhs/100) + b*temps/(c+temps)
        # dp_calc = c*gamma/(b-gamma)
        # spread_calc = temps-dp_calc
        # axx[1].plot(spread_calc, alts)
        
        ''' PLOT smoothed data
        fig2, axx = plt.subplots()
        datas = plot_data.loc[(plot_data['altitude']>17) & (plot_data['altitude']<2100)]
        x_new = datas['altitude']
        f1 = interp1d(altitude, cm_avg)
        f2 = interp1d(alts, rhs)
        xxs, spline = splprep([x_new, datas['relative_humidity']])
        f_spline = splev(spline, xxs)
        #axx.plot(f_spline[1], f_spline[0])
        conv = np.convolve(datas['relative_humidity'], np.ones((10,))/10, mode='valid')
        axx.plot(conv, x_new[:-9])
        axx.grid(b=True, which='both')
        axx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axx.set_xlabel('RH [%]')
        axx.set_ylabel('altitude [m]')
        '''
        # wind_direction = plot_data['wind_from_direction']
        # wind_speed = plot_data['wind_speed']
        
        # nfig, nax = plt.subplots()
        # nax.plot(pressure, alts)
        # nax.set_xlabel('pressure [hPa]')
        # nax.set_ylabel('altitude [m]')
        # nnfig, nnax = plt.subplots()
        # nnax.plot(wind_speed, alts)
        # nnax.set_xlabel('wind speed [m/s]')
        # nnax.set_ylabel('altitude [m]')
        
        plt.show()

run(2021, 12, 7)

# with open('dates_measurements_andoya.txt') as dates:
#     next(dates)
#     for date in dates:
#         split = date.split('.')
#         d = int(str.strip(split[0]))
#         m = int(str.strip(split[1]))
#         y = int('20' + str.strip(split[2])) 
#         if y == 2021 and m >= 7:
#             run(y, m, d, 'andoya')
