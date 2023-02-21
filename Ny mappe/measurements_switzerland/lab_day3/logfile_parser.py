# script for reading and plotting data from day 3 at ETH lab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import re
from scipy import signal 
from utils import *

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

df_iw1 = pd.read_csv('sensor2/LOGFILE.TXT', nrows=2965)
df_iw1 = df_iw1[df_iw1['TIME'] != '#']
df_iw1 = df_iw1[df_iw1['TIME'] != 'TIME']

df_iw2 = pd.read_csv('sensor2/LOGFILE.TXT', skiprows=2966)
df_iw2 = df_iw2[df_iw2['TIME'] != '#']
df_iw2 = df_iw2[df_iw2['TIME'] != 'TIME']

cols1 = ['T', 'RH', 'P']
cols2 = ['T', 'RH', 'P', '#P1', '#P2']
cols3 = ['T', 'RH', 'P', '#P1']

def cf(x):
    try:
        out = float(x)
        return out
    except ValueError as e:
        return np.nan
    
N = 10 # steps to perform running average for
i=1
for df_iw in [df_iw1, df_iw2]:
    df_iw['TIME'] = df_iw['TIME'].apply(lambda x: float(x))
    df_iw['TIME'] = df_iw['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    df_iw[cols3] = df_iw[cols3].applymap(lambda x: np.array(list(map(cf, x.strip().split(" ")))))
    df_iw[cols1] = df_iw[cols1].applymap(np.mean)#, na_action='ignore')
    df_iw['#P1 mean'] = df_iw['#P1'].apply(np.mean)
    df_iw['#P1 sum'] = df_iw['#P1'].apply(np.sum)
    timestamps = fix_timestamps(df_iw)
    df_iw['TIME2'] = timestamps
    
    ra = np.convolve(df_iw['#P1 mean'], np.ones(N)/N, mode='valid')
    df_iw['ra']=0  
    df_iw['ra'].iloc[8:-1]=ra
    i+=1

count_cols1 = process_ops_data(df_iw1)
count_cols2 = process_ops_data(df_iw2)

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)

df_iw1.plot(x='TIME', y='ra', ax=axs[2])

df3 = parse_cpcdata('icewarnTest3_20230201.csv')

df3.plot(x='Time1', y='Concentration1', ax=axs[0])
df3.plot(x='Time2', y='Concentration2', ax=axs[0])
df3.plot(x='Time3', y='Concentration3', ax=axs[0])
df3.plot(x='Time4', y='Concentration4', ax=axs[0])
axs[0].legend(['1 µm', '1 µm', '1 µm', '6 µm'])

df_opc = parse_opcdata('iceWarn_OPC_20230201.CSV')
df_opc.plot(x='Time', y='Count1 (TC)', ax=axs[1])
df_opc.plot(x='Time', y='Count2 (TC)', ax=axs[1])
df_opc.plot(x='Time', y='Count3 (TC)', ax=axs[1])
df_opc.plot(x='Time', y='Count4 (TC)', ax=axs[1])
df_opc.plot(x='Time', y='Count5 (TC)', ax=axs[1])
df_opc.plot(x='Time', y='Count6 (TC)', ax=axs[1])


axs[1].legend([df_opc['Size1'].iloc[0], df_opc['Size2'].iloc[0],
               df_opc['Size3'].iloc[0], df_opc['Size4'].iloc[0],
               df_opc['Size5'].iloc[0], df_opc['Size6'].iloc[0]])

fig1, axs1 = plt.subplots()
df_iw1.plot.scatter(x='TIME2', y='#P1 mean', ax=axs1)
df_iw2.plot.scatter(x='TIME2', y='#P1 mean', ax=axs1)

fig2, axs2 = plt.subplots(nrows=3, ncols=1, sharex=True)

cpc_data1 = df3['Concentration1'].values
cpc_time1 = df3['Time1'].values
cpc_data2 = df3['Concentration2'].values
cpc_time2 = df3['Time2'].values
cpc_data3 = df3['Concentration3'].values
cpc_time3 = df3['Time3'].values
cpc_data4 = df3['Concentration4']
cpc_time4 = df3['Time4'].values
axs2[0].plot(cpc_time1, cpc_data1)
axs2[0].plot(cpc_time2, cpc_data2)
axs2[0].plot(cpc_time3, cpc_data3)
axs2[0].plot(cpc_time4, cpc_data4)
axs2[0].legend(['1 µm', '1 µm', '1 µm', '6 µm'])

opc_data1 = df_opc['Count1 (TC)'].values
opc_data2 = df_opc['Count2 (TC)'].values
opc_data3 = df_opc['Count3 (TC)'].values
opc_data4 = df_opc['Count4 (TC)'].values
opc_data5 = df_opc['Count5 (TC)'].values
opc_data6 = df_opc['Count6 (TC)'].values
opc_time = df_opc['Time']
axs2[1].plot(opc_time, opc_data1)
axs2[1].plot(opc_time, opc_data2)
axs2[1].plot(opc_time, opc_data3)
axs2[1].plot(opc_time, opc_data4)
axs2[1].plot(opc_time, opc_data5)
axs2[1].plot(opc_time, opc_data6)
axs2[1].legend([df_opc['Size1'].iloc[0], df_opc['Size2'].iloc[0],
               df_opc['Size3'].iloc[0], df_opc['Size4'].iloc[0],
               df_opc['Size5'].iloc[0], df_opc['Size6'].iloc[0]])

iw_data1 = df_iw1['ra'].values
iw_time1 = df_iw1['TIME2'].values
iw_data2 = df_iw2['ra'].values
iw_time2 = df_iw2['TIME2'].values
axs2[2].plot(iw_time1, iw_data1)
axs2[2].plot(iw_time2, iw_data2)
#plt1 = df_iw1.plot(x='TIME2', y=count_cols1, ax=axs2[3])
fig_c, ax_c = plt.subplots(nrows=6, ncols=1)
for i, col in zip(range(6), count_cols2): 
    df_iw2.plot(x='TIME2', y=col, ax=ax_c[i])

figp, axp = plt.subplots(nrows=2)
start = datetime.datetime(year=2023, month=2, day=1, hour=17, minute=30, second=0)
stop = datetime.datetime(year=2023, month=2, day=1, hour=17, minute=33, second=0)
mask_TIME2 = (df_iw2['TIME2'] > start) & (df_iw2['TIME2'] <= stop)
mask_Time4 = (df3['Time4'] > start) & (df3['Time4'] <= stop)
data1 = df_iw.loc[mask_TIME2]
data2 = df3.loc[mask_Time4]
axp[0].plot(data1['TIME2'].values, data1['ra'], c='b')
axp[0].set_ylabel("ADC running average", c='b')
ax3 = axp[0].twinx()
ax3.plot(data2['Time4'].values, data2['Concentration4'].values, c='g')
ax3.set_ylabel(r"Concentration (#/cm$^{-3}$)", color="g")
axp[0].set_title('6 µm particles')

start = datetime.datetime(year=2023, month=2, day=1, hour=17, minute=35, second=0)
stop = datetime.datetime(year=2023, month=2, day=1, hour=17, minute=37, second=0)
mask_TIME2 = (df_iw2['TIME2'] > start) & (df_iw2['TIME2'] <= stop)
mask_Time4 = (df3['Time4'] > start) & (df3['Time4'] <= stop)
data1 = df_iw.loc[mask_TIME2]
data2 = df3.loc[mask_Time4]
axp[1].plot(data1['TIME2'].values, data1['ra'], c='b')
axp[1].set_ylabel("ADC running average", c='b')
ax3 = axp[1].twinx()
ax3.plot(data2['Time4'].values, data2['Concentration4'].values, c='g')
ax3.set_ylabel(r"Concentration (#/cm$^{-3}$)", color="g")
axp[1].set_title('6 µm particles')

plt.show()