# script for loading radiosonde and ceilometer data from Lindenberg
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
from scipy.signal import butter, freqz, lfilter
import glob
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

# constants:
g = 9.8076      # m / s^2
c_pd = 1003.5  # J / (kg * K)
H_v = 2501000   # J / kg
R_sd = 287      # J / (kg * K)
R_sw = 461.5    # J /(kg * K)
epsilon = 0.622 

gamma_d = -g / c_pd # dry lapse rate

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


def run(year, month, day):

    y, m, d = date_to_string(year, month, day)
    
    fpaths = glob.glob(f'data/Radiosonde/lindenberg/{y}/{m}/{d}/LIN-RS*.nc')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fpath in fpaths:
            dataset = xr.open_dataset(fpath)    
            data_rs = dataset.to_dataframe()
            data_rs.dropna(inplace=True)
            time = pd.to_datetime(data_rs.index.to_series())
            time = np.array(time.dt.to_pydatetime())
            
            starttime = time[0]
            stoptime = time[-1]
            starttime_string = starttime.strftime('%Y-%m-%d %H:%M:%S')
            stoptime_string = stoptime.strftime('%Y-%m-%d %H:%M:%S')
            
            if starttime.date() == datetime.date(year, month, day):
                _, _, d = date_to_string(year, month, day)
                data_cm = Dataset(f'data/Ceilometer/lindenberg/{y}/{m}/{d}/dataset.nc')
            else:
                _, _, d = date_to_string(year, month, day-1)
                data_cm = Dataset(f'data/Ceilometer/lindenberg/{y}/{m}/{d}/dataset.nc')
            
            pd.set_option('display.max_columns', 100)
            rhs = data_rs['rh'] * 100
            alts = data_rs['alt']
            temps = data_rs['temp'] - 273.15           # temperatures in Celsius

            # Magnus formula 
            b = 18.678
            c = 257.14
            gamma = np.log(rhs/100) + b*temps/(c+temps)
            dp_temps = c*gamma/(b-gamma)
            spread = temps-dp_temps
            ###

            # calculate air density
            R_s = 287.058 # specific gas constant, dry air. Units J/(kg*K)
            pressure = data_rs['press']*100 # plot_data['air_pressure'] is in hPa -> convert to Pa
            rho = pressure/(R_s*(temps + 273.15))
            ##
        
            # plot backscatter and cloud base height from ceilometer
            start_time = data_cm.variables['start_time'][:]
            altitude = data_cm.variables['altitude'][:]
            cloud_base_heights = data_cm.variables['cloud_base_height'][:]
            att_bsc = data_cm.variables['attenuated_backscatter_0'][:] * 1e-6
            station_altitude = data_cm.variables['station_altitude']

            start_time_timestamp = np.array([(t * 60*60*24) for t in start_time])
            start_time_datetime = np.array([datetime.datetime.utcfromtimestamp(t) for t in start_time_timestamp])
            
            indices = np.nonzero( (start_time_datetime > time[0]-datetime.timedelta(minutes=10)) & (start_time_datetime < time[-1]-datetime.timedelta(minutes=5)) )[0]
            ctr = 0
            if (indices[-1] == len(start_time_datetime)-1):
                indices = indices[:-1]
            mat = np.empty((len(altitude), len(indices)))
            cbh_max = np.amax(a=cloud_base_heights[indices, 0])
            try:
                cbh_max = int(np.ceil(cbh_max + station_altitude[0]))
            except:
                print('error caught')
                continue
            if type(cbh_max) != int:
                print(type)
                cbh_max = 1500
            if cbh_max > 2000:
                yticks = np.arange(0,cbh_max+200, 400)
            else:
                yticks = np.arange(0,cbh_max+200, 200)

            ax0 = plt.subplot2grid((2,3), (0,0))
            ax1 = plt.subplot2grid((2,3), (0,1), sharey=ax0)
            ax11 = plt.subplot2grid((2,3), (0,2), sharey=ax0)
            ax2 = plt.subplot2grid((2,3), (1,0), sharey=ax0,colspan=2)
            ax22 = plt.subplot2grid((2,3), (1,2), sharey=ax0)
            ax = [ax0, ax1, ax2, ax11, ax22]
            ax[0].set_ylim(0, cbh_max+200)
            ax[0].set_yticks(yticks)
            ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
            ax[0].plot(rhs, alts)
            ax[0].set_xlim(np.min(rhs)-5, np.max(rhs)+5)
            ax[0].grid(b=True, which='both')
            ax[0].set_ylabel(r'Altitude [m]')
            ax[0].set_xlabel(r'RH [%]')
            ax[0].set_title(f"Radiosonde, {starttime_string}", fontsize=12)
            ids = indices[1:3]
            #for a, i in zip(ax[2::], ids):
            #    a.plot(np.log10(att_bsc[:, i]), altitude)
            #    a.set_xlabel(r'Attenuated backscatter')
            #    a.grid(b=True, which='both')
            #    a.set_title(f"Ceilometer reading, {start_time_datetime[i]}")
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
            ctr=0
            for i in indices:
                for h in cloud_base_heights[i]:
                    # plot a horizontal line representing cloud base layer for each 5 min period
                    x1 = start_time_datetime[i]
                    y1 = h + station_altitude
                    x2 = start_time_datetime[i+1] 
                    y2 = y1
                    ax[2].plot([x1, x2], [y1, y2], color='w', linewidth=1.5)
                mat[:, ctr] = att_bsc[:, i]
                ctr += 1
            #mat = np.ma.masked_where(mat==1.0,mat)
            ax[1].plot(cm_avg[idx2], altitudes)
            #ax[1].plot(rho_m, altitudes, '-.')
            ax[1].grid(b=True, which='both')
            ax[1].set_title(f"Ceilometer, avg {start_time_datetime[ids[0]]} & {start_time_datetime[ids[1]]}", fontsize=12)
            ax[1].set_xlabel(r'Attenuated backscatter [m$^{-1}$ sr$^{-1}$]')

            plot = ax[2].imshow(mat, extent=[matplotlib.dates.date2num(start_time_datetime[indices[0]]), matplotlib.dates.date2num(start_time_datetime[indices[-1]+1]), 123, altitude[-1]], 
                                aspect='auto', origin='lower', cmap='Custom_Cmap', interpolation='none') #vmin=-2.0, vmax=2.0
            ax[2].legend([r'Cloud base'], loc=4, facecolor='k',labelcolor='white')
            ax[2].xaxis.set_tick_params(rotation=45)
            cbar3 = plt.colorbar(plot, ax=ax[2])
            cbar3.set_label(r'Attenuated backscatter')
            ax[2].grid(b=True, which='both')
            ax[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
            ax[2].set_xticks(start_time_datetime[indices])
            ax[2].set_xlim([start_time_datetime[indices][1]], start_time_datetime[indices[4]])
            ax[2].set_xlabel(r'Time [UTC]')
            ax[2].set_ylabel(r'Altitude [m]')
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            cm_avg_d = []
            ds = np.count_nonzero(idx2)
        
            for i in range(ds-2):
                der = (cm_avg[i+1] - cm_avg[i]) / (altitudes[i+1] - altitudes[i]) 
                cm_avg_d.append( der )
            ax[3].plot(cm_avg_d, altitudes[0:ds-2])
            ax[3].grid(b=True, which='both')
            ax[4].plot(temps, alts, label='Measured T')
            #T0 = dataset.attrs['g.SurfaceObs.Temperature'] # temps[0]  Kelvin
            #T0 = float(T0.split(' ')[0])  # surface temperature
            #p0 = dataset.attrs['g.SurfaceObs.Pressure']   # pressure[0]*0.1 
            #p0 = float(p0.split(' ')[0])*0.1   # surface pressure in kPa
            #h0 = dataset.attrs['g.MeasurementSystem.Altitude'] # alts[0]
            #h0 = float(h0.split(' ')[0])
            T0 = temps[0] + 273.15
            p0 = pressure[0] * 0.001
            h0 = alts[0]
            e = 0.61094 * np.exp(17.625*(T0-273.15) / (T0-273.15 + 243.04)) # Magnus equation in C
            r = epsilon * e / (p0 - e)                      # mixing ratio
            # wet lapse rate:
            num = 1.0 + H_v*r/(R_sd*T0)                 # in Kelvin
            den = c_pd + H_v**2*r/(R_sw*T0**2)          # in Kelvin
            gamma_w = -g*num/den
            #print("dry lapse rate: ", gamma_d)
            #print("wet lapse rate: ", gamma_w)
            xs = list(range(2,np.argmax(rhs>=100)+5))
            hs = np.linspace(h0,1000,10)
            T_dry = T0 + gamma_d*(hs - h0) - 273.15
            T_wet = T0 + gamma_w*(hs - h0) - 273.15
            ax[4].plot(T_dry, hs, '--', linewidth=1, label='Dry lapse rate')
            ax[4].plot(T_wet, hs, '--', linewidth=1, label='Wet lapse rate')
            ax[4].legend()
            ax[4].set_xlim(np.min(temps[idx])-2, np.max(temps[idx]+2))
            ax[4].grid(b=True, which='both')
            plt.rcParams.update({'font.size': 14})

            # fig2, axx = plt.subplots(nrows=1, ncols=1, sharey=True)
            # axx.scatter(rhs[xs], temps[xs])
            # plt.gca().invert_yaxis()
            # axx.set_ylabel('Temp [C]')
            # axx.set_xlabel('RH [%]')
            
            ## PLOT smoothed data
            fig2, axx = plt.subplots()
            datas = data_rs.loc[(data_rs['alt']>123) & (data_rs['alt']<2100)]
            x_new = datas['alt']
            #f1 = interp1d(altitude, cm_avg)
            #f2 = interp1d(alts, rhs)
            xxs, spline = splprep([x_new, datas['rh']])
            #f_spline = splev(spline, xxs)
            #axx.plot(f_spline[1], f_spline[0])
            # compute moving average for window of 10 samples:
            conv = np.convolve(datas['rh']*100, np.ones((10,))/10, mode='valid')
            axx.plot(conv, x_new[:-9])
            axx.grid(b=True, which='both')
            axx.yaxis.set_minor_locator(AutoMinorLocator(2))
            axx.set_xlabel('RH [%]')
            axx.set_ylabel('altitude [m]')
            index = np.argmax(rhs)
            rhs_der = []
            for i in range(index+5):
                rhs_der.append( (rhs[i+1] - rhs[i]) / (alts[i+1] - alts[i]) )
            print("alt", alts[index])
            print('rhs_der max = ',np.max(rhs_der[2:]))
            print('max at: ', alts[2:][np.argmax(rhs_der[2:])])
            
            plt.show()
        
run(2020, 8, 14)