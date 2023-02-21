import numpy as np
import pandas as pd
cpcdata = []
newcpc = []

with open('measurements_switzerland\lab\day1\icewarnTest.csv') as f:
    for i, line in enumerate(f):
        cpcdata.append(line)
    f.close()
    Date = cpcdata[4][11:19]
    Start_time = cpcdata[5][11:-3]
    Array=cpcdata[18:]
    n = 0
    while Array[n] != '\n':
        n = n + 1
    Array1 = np.array(Array[0:n])

    Concentration1=[]
    Concentration2=[]
    Concentration3=[]
    CPCtime1=[]
    CPCtime2=[]
    CPCtime3=[]
    for i in range(len(Array1)):
        arr = Array1[i].split(',')
        CPCtime1.append(arr[0]) # time in sec, starting from 1 sec. Already converted counts to #/cc with flow=1lpm
        Concentration1.append(arr[1])  # conc in cm-3
        CPCtime2.append(arr[5]) # time in sec, starting from 1 sec. Already converted counts to #/cc with flow=1lpm
        Concentration2.append(arr[6])  # conc in cm-3
        CPCtime3.append(arr[10]) # time in sec, starting from 1 sec. Already converted counts to #/cc with flow=1lpm
        Concentration3.append(arr[11])  # conc in cm-3
    for i in range(len(CPCtime1)):
        CPCtime1[i] = pd.to_datetime(Date+' '+CPCtime1[i], format="%m/%d/%y %H:%M:%S")#- pd.Timedelta(hours=1)#change to UTC time
    for i in range(len(CPCtime1)):
        CPCtime2[i] = pd.to_datetime(Date+' '+CPCtime2[i], format="%m/%d/%y %H:%M:%S")#- pd.Timedelta(hours=1)#change to UTC time
    for i in range(len(CPCtime1)):
        CPCtime3[i] = pd.to_datetime(Date+' '+CPCtime3[i], format="%m/%d/%y %H:%M:%S")#- pd.Timedelta(hours=1)#change to UTC time
    #a=np.array(CPCtime)
    df = pd.DataFrame
    #Concentration = savgol_filter(Concentration, 3, 2)
    # plt.scatter(CPCtime, Concentration)
    # plt.show()
df = pd.DataFrame()
df['Time1'] = CPCtime1
df['data1'] = Concentration1
df['Time2'] = CPCtime2
df['data2'] = Concentration2
df['Time3'] = CPCtime3
df['data3'] = Concentration3

print(df)