# script for reading and plotting data from day 1 at ETH lab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import re
from scipy import signal
from utils import *

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

l = [i for i in range(0,198)]
l.extend([199, 2291])

df = pd.read_csv('LOGFILE.TXT', skiprows=l)
df.drop(df.tail(1).index, inplace=True)

cols1 = ['T', 'RH', 'P']
cols2 = ['T', 'RH', 'P', '#P1']#, '#P2']
df['TIME'] = df['TIME'].apply(lambda x: float(x))
df['TIME'] = df['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df[cols2] = df[cols2].applymap(lambda x: np.array(list(map(float, x.strip().split(" ")))))

#df['#P1'] = df['#P1'].apply(lambda x: x[x>0])
#df['std']=df['#P1'].map(lambda x: np.mean(x))
#df['mean'] = df['#P1'].map(lambda x: np.std(x))
#df['sum']=np.add(df['std'].values, df['mean'].values)

#col2= ['#P1', '#P2']
#df[['mean1', 'mean2']] = df[col2].applymap(np.mean, na_action='ignore')
#df[['std1', 'std2']] = df[col2].applymap(np.std, na_action='ignore')
#df['sum1'] = df['mean1'] + df['std1']

df[cols1] = df[cols1].applymap(np.mean)#, na_action='ignore')
df['#P1 mean'] = df['#P1'].apply(np.mean)
df['#P1 sum'] = df['#P1'].apply(np.sum)
timestamps = fix_timestamps(df)
df['TIME2'] = timestamps
count_cols = process_ops_data(df)

N = 8 # steps to perform running average for

ra = np.convolve(df['#P1 sum'], np.ones(N)/N, mode='valid')
df['ra']=0  
df['ra'].iloc[N-2:-1]=ra

fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4, ncols=1, sharex=True)
df.plot.scatter(x='TIME', y='P', ax=ax1)
ax1.invert_yaxis()
df.plot.scatter(x='TIME', y='RH', ax=ax2)
df.plot.scatter(x='TIME', y='#P1 mean', c='r', ax=ax3)
df.plot.scatter(x='TIME', y='ra', ax=ax4)
#df.plot.scatter(x='TIME', y='#P2', c='b', ax=ax3)

df2 = pd.read_csv('icewarnTest.csv', skiprows=17)
fig2, [axx1, axx2, axx3] = plt.subplots(nrows=3, ncols=1)
df2.plot(x='Time1', y='Concentration1 (#/cc)', ax=axx1)
df2.plot(x='Time2', y='Concentration2 (#/cc)', ax=axx2)
df2.plot(x='Time3', y='Concentration3 (#/cc)', ax=axx3)

df3=parse_cpcdata('icewarnTest.csv')
print(df3)
df["Concentration1"] = df3['Concentration1']
df["Time1"]=df3['Time1']
df["Concentration2"] = df3['Concentration2']
df["Time2"]=df3['Time2']
df["Concentration3"] = df3['Concentration3']
df["Time3"]=df3['Time3']

# ra1 = np.convolve(df['Concentration1'], np.ones(N)/N, mode='valid')
# ra2 = np.convolve(df['Concentration2'], np.ones(N)/N, mode='valid')
# ra3 = np.convolve(df['Concentration3'], np.ones(N)/N, mode='valid')
# df['ra1']=0  
# df['ra1'].iloc[N-2:-1]=ra1
# df['ra2']=0  
# df['ra2'].iloc[N-2:-1]=ra2
# df['ra3']=0  
# df['ra3'].iloc[N-2:-1]=ra3

fig3, [axs1, axs2, axs3] = plt.subplots(nrows=3, ncols=1, sharex=True)
df.plot(x='TIME2', y='#P1 mean', ax=axs1)
df.plot(x='Time1', y='Concentration1', ax=axs2)
df.plot(x='Time2', y='Concentration2', ax=axs2)
df.plot(x='Time3', y='Concentration3', ax=axs2)
axs2.legend(['200 nm', '400 nm', '600 nm'])
axs2.set_ylabel(r'#/cm$^{-3}$')
axs2.set_xlabel('Time')
df.plot(x='TIME2', y='ra', ax=axs3)

figp, axp = plt.subplots(nrows=2, ncols=1)
start = datetime.datetime(year=2023, month=1, day=30, hour=17, minute=17, second=0)
stop = datetime.datetime(year=2023, month=1, day=30, hour=17, minute=47, second=0)
data = df[df['ra'] < 1000]
# mask_TIME2 = (data['TIME2'] > start) & (data['TIME2'] <= stop)
# mask_Time2 = (data['Time2'] > start) & (data['Time2'] <= stop)
# mask_Time3 = (data['Time3'] > start) & (data['Time3'] <= stop)
data1 = data[data['TIME2'].between(start, stop)]
#data1 = data.loc[mask_TIME2]
data2 = data[data['Time2'].between(start, stop)]
#data2 = data.loc[mask_Time2]
data3 = data[data['Time3'].between(start, stop)]
#data3 = data.loc[mask_Time3]
axp[0].plot(data1['TIME2'].values, data1['ra'], c='b')
axp[0].set_ylabel('ADC running average', c='b')
ax2 = axp[0].twinx()
ax2.plot(data2['Time2'].values, data2['Concentration2'].values, c='g')
ax2.plot(data3['Time3'].values, data3['Concentration3'].values, c='g')
ax2.set_ylabel(r"Concentration (#/cm$^{-3}$)", color="g")
axp[0].set_title('400 nm particles')

b, a = signal.butter(3,0.01)
y = signal.filtfilt(b, a, data1['ra'])
axp[0].plot(data1['TIME2'], y, c='r')

start = datetime.datetime(year=2023, month=1, day=30, hour=17, minute=47, second=0)
stop = datetime.datetime(year=2023, month=1, day=30, hour=18, minute=6, second=0)
# mask_TIME2 = (data['TIME2'] > start) & (data['TIME2'] <= stop)
# mask_Time2 = (data['Time2'] > start) & (data['Time2'] <= stop)
# mask_Time3 = (data['Time3'] > start) & (data['Time3'] <= stop)
# data1 = data.loc[mask_TIME2]
# data2 = data.loc[mask_Time2]
# data3 = data.loc[mask_Time3]

data1 = data[data['TIME2'].between(start, stop)]
data2 = data[data['Time2'].between(start, stop)]
data3 = data[data['Time3'].between(start, stop)]



axp[1].plot(data1['TIME2'].values, data1['ra'], c='b')
axp[1].set_ylabel("ADC running average", c='b')
ax3 = axp[1].twinx()
ax3.plot(data2['Time2'].values, data2['Concentration2'].values, c='g')
ax3.plot(data3['Time3'].values, data3['Concentration3'].values, c='g')
ax3.set_ylabel(r"Concentration (#/cm$^{-3}$)", color="g")
axp[1].set_title('600 nm particles')

# try filtering signal; lowpass / forward-backward pass
b, a = signal.butter(3,0.01)
y = signal.filtfilt(b, a, data1['ra'])
axp[1].plot(data1['TIME2'], y, c='r')

plt.show()