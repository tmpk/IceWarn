import glob
from sys import int_info
from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import E
import sklearn.metrics
import sklearn.feature_selection
import os
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import matplotlib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def objective_function(x, a, b):
    return a*(1 - x/100.1) ** b

def detect_inversion(df):
    # Finds the rows in df where an inversion exists (i.e temperature increases wih 
    # height) and returns it in the form of a boolean array 
    
    temps = df['temp']
    temps_diff = np.diff(temps, prepend=temps.iloc[0])
    bool_array = temps_diff > 0
    
    return bool_array

def get_interval_idx(inv_ids):
    # helper function for finding the indices defining intervals where 
    # the lapse rate is negative
    inv_idx = np.where(inv_ids == True)[0] # indices where temp increases  
    subsets_idx = []                       # array with (start, stop) tuple defining interval where temp decreases
    start = 0
    if len(inv_idx) == 0:
        subsets_idx.append( (0, -1) )
    else:
        for ex in inv_idx:
            last = (ex == inv_idx[-1])
            if ex == start:
                if last:
                    subsets_idx.append( (ex+1, -1) )
                else:
                    start = ex+1
                    continue
            else:
                if last:
                    subsets_idx.append( (start, ex) )
                    subsets_idx.append( (ex+1, -1) )
                    #subsets_idx.append( (start, -1) )
                else:
                    subsets_idx.append( (start, ex) )
                    start = ex+1
    return subsets_idx

def calc_means(location, df, colname, subsets_idx):
    # calculate means of dataframe column subsets where lapse rate is negative
    # subsets of three or less measurements are ignored for andoya; 6 for lindenberg 
    # - this due to differences in sampling rates
    
    means = []
    
    limit = 3 if location == 'andoya' else 6
    for subset in subsets_idx:
        start = subset[0]
        stop = subset[1]
        if (subset[1] -  subset[0]) <= limit and not (subset[1] == -1):
            continue
        mean = np.mean( df[colname].iloc[ start:stop ])
        if np.isnan(mean): 
            continue
        else:
            means.append( mean )    

    return means

def calc_stds(location, df, colname, subsets_idx):
    # calculate standard deviations of dataframe column subsets where 
    # lapse rate is negative
    # subsets of three or less measurements are ignored for andoya; 6 for lindenberg
    stds = []
    limit = 3 if location == 'andoya' else 6
    for subset in subsets_idx:
        if (subset[1] -  subset[0]) <= limit and not (subset[1] == -1):
            continue
        stds.append( np.mean( df[colname].iloc[ subset[0]:subset[1] ]) )
    return stds

def calculate_dewpoint(df):
    # Magnus formula 
    b = 18.678
    c = 257.14
    rhs = df['rh']
    temps = df['temp']
    gamma = np.log(rhs/100) + b*temps/(c+temps)
    dp_calc = c*gamma/(b-gamma)
    dp_depr = temps - dp_calc
    return dp_depr

def calculate_vp(df):
    # Calculates vapor pressure from RH and T measurements, and Magnus/Buck equation
    rhs = df['rh']
    temps = df['temp']
    #svp = 6.1094 * np.exp(17.625 * temps / (temps + 243.04)) # saturation vapor pressure (Magnus eq.)
    svp = 6.1121 * np.exp( ( 18.678 - temps / 234.5) * (temps / (257.14 + temps))) # Buck eq.
    vp = rhs / 100 * svp
    return svp, vp

def estimate_rh(rh, svp, p): 
    # estimates RH from initial RH measurement, satuation vapor pressures, 
    # air pressures. Adds noise to estimate for simulation purposes
    RH_est = np.array([])
    for i in range(0, len(rh), 10):
        rh0 = rh.iloc[i]
        noise = noise = np.random.normal(0,3.5/3, 1)
        y = (rh0 + noise)*svp.iloc[ i ] / p.iloc[ i ]
        temp = y * p.iloc[i:i+10] / svp.iloc[i:i+10]
        RH_est = np.append(RH_est, temp)
    RH_est = np.add(RH_est, noise)

    return RH_est

def plot_measurement_series(df, fpath): 
    #fig, ax = plt.subplots(nrows=1, ncols=3, sharey='all', sharex='col')
    fig = plt.figure()
    ax0 = plt.subplot(141)
    ax1 = plt.subplot(142, sharey=ax0)
    ax2 = plt.subplot(143)#, sharey=ax0)
    ax3 = plt.subplot(144)

    df_temp = df['temp']
    df_rh = df['rh']
    df_alt = df.index.to_numpy()
    df_bsc = df.dropna(subset=['att_bsc'])
    df_svp = df['saturation_vapor_pressure']
    df_p = df['pressure']

    ### curve fitting
    # y = df_bsc['rh'].values
    # x = df_bsc['att_bsc'].values
    # popt, _ = curve_fit(objective_function, y, x)
    # print(popt)
    # ax3.plot(objective_function(y, *popt), y)
    ###

    idx80 = get_nearest(80, df_bsc, 'rh')
    bsc0 = df_bsc['att_bsc'].loc[idx80]
    df_bsc_grad = df['att_bsc_grad']
    df_pot_temp = df['pot_temp']
    inv_ids = detect_inversion(df)
    ax0.plot(df_temp, df_alt)
    ax0.scatter(df['temp'].loc[inv_ids], df_alt[inv_ids], marker = 'x')
    ax0.plot(df_pot_temp, df_alt)
    ax0.scatter(df_pot_temp.loc[inv_ids], df_alt[inv_ids], marker = 'x')
    ax0.legend(['temperature', 'potential temperature'])
    ax0.set_title('temperature')
    ax0.set_ylabel('Altitude [m]')
    ax0.set_xlabel(r'Temperature [$^\circ$C]')
    ax1.set_xlabel(r'Relative humidity [%] ')

    ax1.plot(df_rh, df_alt)

    subsets_idx = get_interval_idx(inv_ids)
    for subset in subsets_idx:
            #ax1.scatter(df_rh.iloc[ subset[0]:subset[1] ], df_alt[ subset[0]:subset[1] ])
            RH = df_rh.iloc[ subset[0]:subset[1] ]
            pot_temp = df_pot_temp.iloc[ subset[0]:subset[1]]
            svp = df_svp.iloc[  subset[0]:subset[1]  ]
            p = df_p.iloc[  subset[0]:subset[1]  ]
            alt = df_alt[ subset[0]:subset[1] ]
            #RH_est = estimate_rh(RH, svp, p, alt)
            #ax1.plot(RH_est, alt, color='r')

            ##--- 
            rh0 = RH.iloc[0]
            yv = rh0*svp.iloc[ 0 ] / p.iloc[ 0 ]
            rh_est = yv * p / svp
            ax1.plot(rh_est, alt, color='g', linestyle='--')
            
            # f, axs = plt.subplots(ncols=2, nrows=1, sharey=True)
            # a1 = (RH.iloc[-1] - RH.iloc[0]) / (alt[-1] - alt[0])
            # b1 = RH.iloc[0] - a1*alt[0]
            # y1 = a1*alt + b1
            # #ax1.plot(y1, alt, color='c', linestyle='--')
            # axs[1].plot(y1-RH, alt, color='c', linestyle='--')
            # a2 = (pot_temp.iloc[-1] - pot_temp.iloc[0]) / (alt[-1] - alt[0])
            # b2 = pot_temp.iloc[0] - a2*alt[0]
            # y2 = a2*alt + b2
            # axs[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2e-2))
            # axs[0].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1e-2))
            # axs[0].plot(y2-pot_temp, alt, color='b')
            # print("correlation: ", np.corrcoef(pot_temp-y2, RH-y1))
            ##---------

    #smoothed_rh = np.convolve(df_rh, np.ones(5)/5, mode='valid')
    #ax1.plot(smoothed_rh, df_alt[4::], linestyle='dashed')
    #ax1.scatter(df_rh.loc[inv_ids], df_alt[inv_ids], marker='x')
    ax1.legend(['measured', 'estimated'])
    ax1.set_title('humidity')
    
    # -----
    alts = df_bsc.index.to_numpy()
    ax2.scatter(df_bsc['att_bsc'], alts)
    ax2.set_title('backscatter')
    ax2.set_ylabel('altitude')
    bsc_diff = np.diff(df_bsc['att_bsc'])
    alts_diff = np.diff(alts)
    bsc_grad = bsc_diff/alts_diff
    ax3.scatter(bsc_grad, alts[1::])
    #ax3.scatter(df_bsc['att_bsc'], df_bsc['rh']) # plots bsc aaf. rh
    ax3.set_title('att. bsc aaf. RH')
    ax3.set_ylabel('RH')
    #ax3.invert_yaxis()
    ax2.grid(b=True, which='both')
    ax3.grid(b=True, which='both')
    loc = str.rsplit(fpath, "\\")[-2]
    string = str.rsplit(fpath, "\\")[-1]
    string = string.rsplit("_")
    date = string[0].rsplit("-")
    y = date[0]
    m = date[1]
    d = date[2]
    t = string[1].rsplit(".")[0].replace("-", ":")
    plt.suptitle(f"{loc}, {d}.{m}.{y} {t}")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    return fig, [ax0, ax1, ax2, ax3]

def get_nearest(value, df, colname):
    # finds the entry in df[colname] that is closest to "value"
    # and returns the index

    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:

        lower = df[colname][df[colname] <= value]
        upper = df[colname][df[colname] > value]

        if lower.empty and upper.empty:
             return np.nan 

        if not lower.empty:
            lower = lower.idxmax()
            lower_diff = np.abs(value - df[colname].loc[lower])
        else:
            lower_diff = np.inf

        if not upper.empty:
            upper = upper.idxmin()
            upper_diff = np.abs(value - df[colname].loc[upper])
        else:
            upper_diff = np.inf
        
        return lower if lower_diff < upper_diff else upper

def calc_gradients(df):
    # approximates grads of temp, rh aaf. of altitude

    T = df['temp']
    RH = df['rh']
    alt = df.index

    T_diff = np.diff(T)
    RH_diff = np.diff(RH)
    alt_diff = np.diff(alt)
    T_grad = T_diff / alt_diff
    
    RH_grad = RH_diff / alt_diff
    T_grad_max_idx = np.nanargmax(T_grad)
    T_grad_max = T_grad[T_grad_max_idx]
    RH_grad_max_idx = np.nanargmax(RH_grad)
    RH_grad_max = RH_grad[RH_grad_max_idx]
    T_grad_min_idx = np.nanargmin(T_grad)
    T_grad_min = T_grad[T_grad_min_idx]
    RH_grad_min_idx = np.nanargmin(RH_grad)
    RH_grad_min = RH_grad[RH_grad_min_idx]
    RH_at_grad_max = df['rh'].iloc[RH_grad_max_idx]
    #print("max dT/dh: ", T_grad_max*1000, 'C / km')
    #print("max dRH/dh: ", RH_grad_max, '%RH / m')
    #print("min dT/dh: ", T_grad_min*1000, 'C / km')
    #print("min dRH/dh: ", RH_grad_min, '%RH / m')

    return T_grad_max, T_grad_min, RH_grad_max, RH_grad_min, RH_at_grad_max

def delete_file(fpath):
    os.remove(fpath)
    print("file: ", fpath, " deleted.")

def smooth_ma(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = ( cumsum_vec[window_width:] - cumsum_vec[:-window_width] ) / window_width
    return ma_vec

def interpolate(df, col):
    # Linearly interpolates values in column 'col' in dataframe 'df'
    # (inside and forward interpolation)
    nans = df[col].isna().values
    df2 = df.copy(deep=True)
    i = 0
    interval = False
    for elem in nans[1:]:
        if interval == False and nans[i] == True:   # block handling case with leading nans
            i+=1
            continue
        if elem != nans[i]:
            if interval == False: # at start of interpolation interval
                start_idx = i 
                interval = True
            else:                 # at stop of interpolation interval
                stop_idx = i+1
                interval = False
                p0 = df2[col].iloc[start_idx]
                h0 = df2.index[start_idx]
                p1 = df2[col].iloc[stop_idx]
                h1 = df2.index[stop_idx]
                delta = (p1-p0)/(h1-h0)
                interpolated = p0 + np.multiply(df2.index.values[ start_idx+1:stop_idx ]-h0, delta)
                df2[col].iloc[ (start_idx+1):stop_idx ] = interpolated
        i += 1
    return df2[col]

def compute_mi(df, columns, target):
    X = df[columns].values 
    y = df[target].values
    MI1 = sklearn.feature_selection.mutual_info_regression(X, y)    
    print("mutual information: ", MI1)
                                                          

locations = ['andoya', 'lindenberg']
out_rh_max  = []
out_bsc_max = []
out_rh = []
out_bsc = []
out_bsc_grad = []
out_rh_grad = []
out_temp_grad = [] 
# mean_bscmax_andoya = 0.00012307412571805495
# std_bscmax_andoya = 8.294057494526954e-05
# mean_bscmax_lindenberg = 0.00010307915752111754
# std_bscmax_lindenberg = 8.843119530188132e-05
bsc_max_anx = []
bsc_max_ldb = []
temp_means_out = []
temp_stds_out = []
rh_means_out = []
rh_stds_out = []

for location in locations:
    fpaths = glob.glob(f"data\selected_data\\{location}\*")
    # converter function to handle nan values in loaded data
    converter_function = lambda x: np.nan if x =='-' else float(x)
    dict_convert = {1: converter_function, 2: converter_function, 3: converter_function, 4: converter_function, 5: converter_function}

    bsc_all = []
    bsc_max_all = []
    rhs_max_all = []
    rhs_all = []
    bsc_grad_all = []
    temp_grad_all = []
    rh_grad_all = []
    temp_means_all = []
    temp_stds_all = []
    rh_means_all = []
    rh_std_all = []

    for fpath in fpaths:
        df = pd.read_csv(fpath, index_col=0, converters=dict_convert)
        df.replace(to_replace='-', value=np.nan, inplace=True)
        df.set_index('alt', inplace=True)
        #if location == 'andoya':
        #    ...
            #df['att_bsc'] = (df['att_bsc'] - mean_bscmax_andoya ) / std_bscmax_andoya
        #else:
        #    ...#df['att_bsc'] = (df['att_bsc'] - mean_bscmax_lindenberg ) / std_bscmax_lindenberg
        
        # interpolate values in table using "forward" and "inside" method,
        df['rh'] = df['rh'].interpolate(method='values', limit_direction='forward', axis=0, limit_area='inside')
        df['temp'] = df['temp'].interpolate(method='values', limit_direction='forward', axis=0, limit_area='inside')
        df['att_bsc_interpolated'] =  interpolate(df, 'att_bsc') # df['att_bsc'].interpolate(method='index', limit_direction='forward', axis=0, limit_area='inside')
        #df['att_bsc'] = df['att_bsc'] / df['att_bsc'].max()
        df['dew_point'] = calculate_dewpoint(df)
        svp, vp = calculate_vp(df)
        df['saturation_vapor_pressure'] = svp
        df['vapor_pressure'] = vp
        df['pressure'] = df['pressure'].interpolate(method='values', limit_direction='forward', axis=0, limit_area='inside')
        df['pot_temp'] = ( df['temp'] + 273.15)*(1000 / df['pressure'] )**0.286 - 273.15 # potential temperature
        df['alt_diff'] = np.diff(df.index, prepend=np.nan)
        temp_diffs = np.diff(df['temp'], prepend=np.nan)
        pot_temps_diffs = np.diff(df['pot_temp'], prepend=np.nan)
        rhs_diffs = np.diff(df['rh'], prepend=np.nan)
        df['att_bsc_diff'] = np.diff(df['att_bsc_interpolated'], prepend=np.nan)
        df['temps_grad'] = temp_diffs / df['alt_diff']
        df['pot_temps_grad'] = pot_temps_diffs/df['alt_diff']
        df['rhs_grad'] = rhs_diffs/df['alt_diff']
        df['att_bsc_grad'] = df['att_bsc_diff'] / df['alt_diff']
        #df.dropna(inplace=True, subset=['rh', 'att_bsc', 'att_bsc_grad'])
        stop_idx = df['att_bsc'].idxmax()
        df2 = df.loc[:stop_idx, :].copy(deep=True)
        #df2 = df2[df2['rh'] > 90]
        #if df2['att_bsc'][df2['att_bsc'] < 0].any():
            
        df2['grad_inc'] = np.nan
        vec1 = df2['att_bsc_grad'].iloc[1:].values
        vec2 = df2['att_bsc_grad'].iloc[0:-1].values
        df2['grad_inc'].iloc[1:] = np.divide(vec1, vec2)

        #df2['att_bsc'] = df2['att_bsc'] + 1e-5
        #df2['att_bsc'] = np.log10(df2['att_bsc'])
        df_nona = df2.dropna(subset=['att_bsc'])
    
        rh_out = df_nona['rh']
        bsc_out = df_nona['att_bsc']
        bsc_grad_out = df_nona['att_bsc_grad']
        
        temps_grad_out = df2['temps_grad']
        rhs_grad_out = df2['rhs_grad']

        idx_max = np.argmax(df2['att_bsc'])
        bsc_max_out = df2['att_bsc'].iloc[idx_max]
        rhs_max_out = df2['rh'].iloc[idx_max]
        ####
        inv_ids = detect_inversion(df2)
        subsets_idx = get_interval_idx(inv_ids)
        rhs_means = calc_means(location,  df2, 'rhs_grad', subsets_idx)
        rhs_stds = calc_stds(location,  df2, 'rhs_grad', subsets_idx)
        temp_means = calc_means(location,  df2, 'temps_grad', subsets_idx )
        temp_stds = calc_stds(location,  df2, 'temps_grad', subsets_idx)
        temp_means_all.extend(temp_means)
        temp_stds_all.extend(temp_stds)
        rh_means_all.extend(rhs_means)
        rh_std_all.extend(rhs_stds)

        ####
        bsc_max_all.append(bsc_max_out)
        rhs_max_all.append(rhs_max_out)
        rhs_all.extend(rh_out)
        bsc_all.extend(bsc_out)
        bsc_grad_all.extend(bsc_grad_out)
        for idx in subsets_idx:
            start = idx[0]
            stop = idx[1]
            rh_grad_all.extend(rhs_grad_out.iloc[start:stop])
            temp_grad_all.extend(temps_grad_out.iloc[start:stop])
        

        # find mutual information
        cols = ['temp', 'pot_temp', 'temps_grad', 'pot_temps_grad', 'rhs_grad']
        target = 'rh'
        data = df2.dropna(subset=['temps_grad'])
        data = data[data['rh'] > 80]
        #compute_mi(data, cols, target)
        cols = ['att_bsc', 'att_bsc_grad']
        target = 'rh'
        data = df2.dropna(subset=['att_bsc', 'att_bsc_grad'])
        data = data[data['rh'] > 90]
        #compute_mi(data, cols, target)

        #plot_measurement_series(df2, fpath)
        #plt.show()
        # /end mutual information

        #if df['att_bsc'].isna().any():
        #     print(df['att_bsc'])
        #    plot_measurement_series(df2, fpath)
        
    out_rh_max.append(rhs_max_all)
    out_bsc_max.append(bsc_max_all)
    out_rh.append(rhs_all)
    out_bsc.append(bsc_all)
    out_bsc_grad.append(bsc_grad_all)
    out_rh_grad.append(rh_grad_all)
    out_temp_grad.append(temp_grad_all)
    temp_means_out.append(temp_means_all)
    temp_stds_out.append(temp_stds_all)
    rh_means_out.append(rh_means_all)
    rh_stds_out.append(rh_std_all)

f, a = plt.subplots(ncols=2, nrows=1, sharex=True) #, sharex=True, sharey=True)
#a[0].scatter(out_bsc[0], out_rh[0], marker='.')
#a[1].scatter(out_bsc[1], out_rh[1], marker='.')
x1 = np.array(out_bsc[0])
y1 = np.array(out_rh[0])
x2 = np.array(out_bsc[1])
y2 = np.array(out_rh[1])
binns = np.linspace(0,100,101)
idxs1 = np.digitize(y1, binns)
idxs2 = np.digitize(y2, binns)
df_out1 = pd.DataFrame({'rh': binns[idxs1-1], 'att_bsc': x1})
df_out1.sort_values(by='rh', inplace=True)
df_out_grouped1 = df_out1.groupby('rh')
df_out2 = pd.DataFrame({'rh': binns[idxs2-1], 'att_bsc': x2})
df_out2.sort_values(by='rh', inplace=True)
df_out_grouped2 = df_out2.groupby('rh')

lst1 = {}
for group in df_out_grouped1.groups:
    g = df_out_grouped1.get_group(group)
    lst1[f"{group}"] = g['att_bsc'].values

lst2 = {}
for group in df_out_grouped2.groups:
    g = df_out_grouped2.get_group(group)
    lst2[f"{group}"] = g['att_bsc'].values

df_anx = pd.DataFrame.from_dict(lst1, orient='index')

for row in df_anx.iterrows():
    rh = row[0]
    bsc = row[1].dropna()
    a[0].scatter(np.mean(bsc), rh, color='b')
#df_anx.to_csv(f'data\selected_data\\means\means_anx.csv')

df_lin = pd.DataFrame.from_dict(lst2, orient='index')
for row in df_lin.iterrows():
    rh = row[0]
    bsc = row[1].dropna()
    a[1].scatter(np.mean(bsc), rh, color='b')
#df_lin.to_csv(f'data\selected_data\\means\means_lin.csv')

loc = matplotlib.ticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
a[0].yaxis.set_major_locator(loc)
a[1].yaxis.set_major_locator(loc)
######################
f1, a1=plt.subplots(ncols=2, nrows=1)
a1[0].scatter(out_bsc_grad[0], out_rh[0], marker='.') #a1[0].scatter(df_out1['att_bsc_grad'], df_out1['rh'], marker='.')
a1[1].scatter(x1, binns[idxs1-1])

# Calculate the point density
# xy1 = np.vstack([x1, y1])
# z1 = gaussian_kde(xy1)(xy1)
# # Sort the points by density, so that the densest points are plotted last
# idx = z1.argsort()
# x1, y1, z1 = x1[idx], y1[idx], z1[idx]
# a[0].scatter(x1, y1, c=z1)#, c=z1, s=50, edgecolor='') #

## correlation
# fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
# x1 = np.array(rh_means_out[0])
# y1 = np.array(temp_means_out[0])
# x2 = np.array(rh_means_out[1])
# y2 = np.array(temp_means_out[1])
# xy1 = np.vstack([x1, y1])
# xy2 = np.vstack([x2, y2])
# z1 = gaussian_kde(xy1)(xy1)
# z2 = gaussian_kde(xy2)(xy2)
# idx1 = z1.argsort()
# idx2 = z2.argsort()
# x1, y1, z1 = x1[idx1], y1[idx1], z1[idx1]
# x2, y2, z2 = x2[idx2], y2[idx2], z2[idx2]
# ax[0].scatter( x1, y1, c=z1, marker='.') # out_rh_grad[0], out_temp_grad[0], marker='.')
# ax[0].ticklabel_format(axis='x', style='sci', scilimits=(-5,-5))
# ax[1].scatter(x2, y2, c=z2, marker='.') #out_rh_grad[1], out_temp_grad[1], marker='.')

# for a in ax:
#     a.spines['top'].set_color('none')
#     a.spines['left'].set_position('zero')
#     a.spines['right'].set_color('none')
#     a.spines['bottom'].set_position('zero')
#     a.annotate('d(RH)/dh', xy=(1, 0), xycoords=('axes fraction', 'data'), 
#     xytext=(10, -10), textcoords='offset points',
#     ha='center', va='center',
#     )
#     a.annotate('d(T)/dh', xy=(0, 1), xycoords=('data', 'axes fraction'), 
#     xytext=(0, 10), textcoords='offset points',
#     ha='center', va='center',
#     )
#     yabs_max = abs(max(a.get_ylim(), key=abs))
#     a.set_ylim(ymin=-yabs_max, ymax=yabs_max)
#     xabs_max = abs(max(a.get_xlim(), key=abs))
#     a.set_xlim(xmin=-xabs_max, xmax=xabs_max)
#     a.grid(False)

# ax[1].set_title('lindenberg', pad=30)
# ax[0].set_title('andoya', pad=30)

# print("correlation andoya: ", np.corrcoef(out_rh_grad[0], out_temp_grad[0]))
# print("correlation lindenberg: ", np.corrcoef(out_rh_grad[1], out_temp_grad[1]))
## end correlation

# Mutual information
# X1 = np.transpose( np.array( [ out_bsc[0], out_bsc_grad[0] ] ) )
# y1 = np.array(out_rh[0])
# print(X1)
# print(y1)
# MI1 = sklearn.feature_selection.mutual_info_regression(X1, y1)                                                              
# print("mutual information andoya: ", MI1)
# X2 = np.transpose( np.array( [ out_bsc[1], out_bsc_grad[1] ] ) )
# y2 = np.array(out_rh[1])
# MI2 = sklearn.feature_selection.mutual_info_regression(X2, y2)
# print("mutual information lindenberg: ", MI2)
# end mutual information

fig2, ax2 = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True)
bin1 = np.linspace(-0.2, 0.2, 100)
bin2 = np.linspace(-0.2, 0.2, 100)
ax2[0].hist(out_bsc_grad[0])#, bins=bin1)#, density=True)
ax2[0].set_title('andoya')
ax2[1].hist(out_bsc_grad[1])#, bins=bin2)#, density=True)
ax2[1].set_title('lindenberg')
ax2[0].set_ylabel('Frequency')
ax2[0].set_xlabel(r'$d(\beta)/dz$')
ax2[1].set_xlabel(r'$d(\beta)/dz$')
#ax2[0].set_xticks(bin)
#ax2[0].ticklabel_format(style='sci', axis='x', scilimits=(-5,-5))

plt.show()