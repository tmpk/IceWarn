# Script for plotting backscatter coefficients of different ensembles of aerosols
# as a function of relative humidtity. 
# Data plotted includes coefficients calculated from Mie theory using the software package MOPSMAP
# and empirical measurements 

from netCDF4 import Dataset 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import curve_fit

def func(x,a,b):
    return (a + x/(100 - x) ) ** b

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
plt.suptitle('Normalized backscatter coefficient aaf. of relative humidity', fontsize=12)

fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True)
filenames_lin = glob.glob('aerosol_dist/lin/*.txt')
filenames_anx = glob.glob('aerosol_dist/anx/*.txt')
filenames_oth = glob.glob('aerosol_dist/all/*.txt')
lgnds = [[], []]
n=0
fxs, axs = plt.subplots()
legendx = []
plt.suptitle('Backscatter coefficient aaf. of relative humidity', fontsize=12)
for fn in filenames_oth:
    lgnd = fn.split("\\")[-1]
    lgnd = lgnd.split("-")[0]
    legendx.append(lgnd)
    
    data = np.genfromtxt(fn, dtype=float)
    RH = data[0]
    bsc = data[1]
    bsc0 = bsc[0]
    axs.plot(-1/bsc, RH)
axs.legend(legendx, prop={'size': 8}, loc='upper left')
axs.grid(which='both')
axs.set_ylabel('Relative humidity [%]', fontsize=10)
axs.set_xlabel(r'Backscatter coefficient [$-1/\beta$]', fontsize=10)

for filenames in [filenames_anx, filenames_lin]:
    for fn in filenames:
        lgnd = fn.split("\\")[-1]
        lgnd = lgnd.split("-")[0]
        lgnds[n].append(lgnd)
        
        data = np.genfromtxt(fn, dtype=float)
        RH = data[0]
        bsc = data[1]
        bsc0 = bsc[0]
        ax[n].plot(bsc/bsc0, RH)
        #ax1[n].set_xscale("log")
        ax1[n].plot(bsc, RH)

        #ax1[n].set_xscale("log")
        #if lgnd == 'continental_polluted':
        #    popt, pcov = curve_fit(func, RH, bsc)
        #     print(popt)
        #    ax1[n].plot(func(RH, *popt), RH)
        #    lgnds[n].append('continental_polluted_fit')
    n+=1

for loc, axx in zip(['Andoya', 'Lindenberg'], ax):
    axx.set_title(loc, fontsize=10)
    #axx.xaxis.set_major_locator(MultipleLocator(5))
    #axx.xaxis.set_minor_locator(MultipleLocator(1))
    axx.yaxis.set_major_locator(MultipleLocator(5))
    axx.yaxis.set_minor_locator(MultipleLocator(1))
    axx.grid(which='both')
    axx.set_ylabel('Relative humidity [%]', fontsize=10)
    axx.set_xlabel(r'Backscatter coefficient [$-1/\beta$]', fontsize=10)
ax[1].yaxis.set_tick_params(labelleft=True)

plt.suptitle('Backscatter coefficient aaf. of relative humidity', fontsize=12)
for loc, axx in zip(['Andoya', 'Lindenberg'], ax1):
    axx.set_title(loc, fontsize=10)
    # axx.xaxis.set_major_locator(MultipleLocator(1e-5))
    # axx.xaxis.set_minor_locator(MultipleLocator(0.5e-5))
    # axx.yaxis.set_major_locator(MultipleLocator(5))
    # axx.yaxis.set_minor_locator(MultipleLocator(1))
    axx.grid(which='both')
    axx.set_ylabel('Relative humidity [%]', fontsize=10)
    axx.set_xlabel(r'Backscatter coefficient [$-1/\beta$]', fontsize=10)
ax1[1].yaxis.set_tick_params(labelleft=True)
ax1[0].set_ylim(bottom=50)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)

# plot empirical measurements
# converter function to handle nan values in loaded data
converter_function = lambda x: np.nan if x =='-' else float(x)
dict_convert = {1: converter_function, 2: converter_function, 3: converter_function, 4: converter_function, 5: converter_function}

fpaths = ['data\selected_data/andoya/2019-12-19_02-05-00.csv', 
          'data\selected_data/andoya/2020-05-30_11-03-00.csv']
means_anx = pd.read_csv('data\selected_data/means/means_anx.csv')
means_lin = pd.read_csv('data\selected_data/means/means_lin.csv')

fig7, ax7 = plt.subplots(nrows=1, ncols=2,sharex=True)
n = 0
col = 'b'
mar = '.'
mean_bscs = [[], []]
mean_bsc0 = [np.nan, np.nan]
bscs = [[], []]
rhs = [[], []]
for means in [means_anx, means_lin]:
    for i in range(0, len(means.index)): # fpath in fpaths:
            #df = pd.read_csv(fpath, index_col=0, converters=dict_convert)
            #df.replace(to_replace='-', value=np.nan, inplace=True)
            #df.set_index('alt', inplace=True)
            #df['rh'] = df['rh'].interpolate(method='values', limit_direction='forward', axis=0, limit_area='inside')
            #df_bsc = df.dropna(subset=['att_bsc'])
            #df_bsc = df_bsc[df_bsc['rh'] <= 99] 
            #df_bsc['att_bsc'] *= 10**-1
            #ax1.scatter(df_bsc['att_bsc'], df_bsc['rh'], marker=".")
            rh = means.iloc[i][0]
            if rh < 80 or rh > 99: continue

            rhs[n].append(rh)
            bsc = means.iloc[i][1::].dropna()
            bscs[n].append(bsc)
            mean_bsc = np.median(bsc)
            mean_bscs[n].append(mean_bsc)
            if rh == 50:
                mean_bsc0[n] = mean_bsc
            if rh >= 50:
                ax[n].scatter(mean_bsc/mean_bsc0[n], rh, color=col, marker='.')
            ax1[n].scatter(mean_bsc, rh, color=col, marker=mar)
            #l = fpath.rsplit('/')[-1]
            #l = l.rsplit('.')[0]
            #l = l.replace("_", " ")
            #l = l.rsplit(" ")[0] + " " + l.rsplit(" ")[-1].replace("-", ":")
            #lgnd.append(l)
    #ax7[n].xaxis.set_major_locator(MultipleLocator(2e-5))
    ax7[n].boxplot(bscs[n], vert=False, showfliers=False)
    ax7[n].set_ylabel(r'Relative humidity [%]', fontsize=8)
    ax7[n].set_xlabel(r'Backscatter coefficient [$-1/\beta$]', fontsize=8)
    ax7[n].yaxis.set_tick_params(labelleft=True)
    ax7[n].set_title(loc)
    ax7[n].grid(which='both')
    arr = -1/np.array(mean_bscs[n])
    x = np.array(rhs[n])
    
    p = np.polyfit(x, arr, deg=2)
    y = np.poly1d(p)
    if n==0:
        axs.plot(y(rhs[n]), rhs[n], linestyle='--', color='m')
    else:
        axs.plot(y(rhs[n]), rhs[n], linestyle='--', color='c')
    n += 1
    col = 'r'
    mar = '.'

legendx.append('curve fit emp. data anx')
legendx.append('curve fit emp. data lin')
axs.legend(legendx)

plt.show()

ax7[0].set_title('Andoya')
ax7[1].set_title('Lindenberg')
ax7[0].set_ylim(ymin=26)
ax7[1].set_ylim(ymin=46)

lgnds[0].append('mean emp. value')
lgnds[1].append('mean emp. value')
ax[0].legend(lgnds[0], prop={'size': 8}, loc='upper left')
ax[1].legend(lgnds[1], prop={'size': 8}, loc='upper left')
ax1[0].legend(lgnds[0], prop={'size': 8}, loc='upper left')
ax1[1].legend(lgnds[1], prop={'size': 8}, loc='upper left')

labels1 = [str(x) for x in np.arange(60,100,1)]
labels2 = [str(x) for x in np.arange(60,100,1)]
ax7[0].set_yticklabels(labels1)
ax7[1].set_yticklabels(labels2)

#plt.yticks(ax7[1].get_yticks(), labels2)

rhs = np.array(rhs[0:-1])
mean_bscs = np.array(mean_bscs[0:-1])

#popt, pcov = curve_fit(func, rhs, mean_bscs)
#ax1[1].plot(func(rhs, *popt), rhs)
#a, b, = [1.66764713e-06, 2.16351642e-07]
#ax1[1].plot(func(rhs, a, 1.5e-01), rhs)

plt.show()