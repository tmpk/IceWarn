
def parse_cpcdata(filepath): 
    cpcdata = []

    with open(filepath) as f:
        for i, line in enumerate(f):
            cpcdata.append(line)
        f.close()
        Date = cpcdata[4][11:19]
        Start_time = cpcdata[5][11:-3]
        Array=cpcdata[18:]

        # find number of samples in file
        tmp = cpcdata[17].split(',')
        nSamples = 0
        for elem in tmp:
            if re.search('Concentration*', elem):
                nSamples += 1
        ###

        Concentrations = []
        CPCtimes = []
        for i in range(0,nSamples):
            Concentrations.append([])
            CPCtimes.append([])

        n = 0
        while Array[n] != '\n':
            n = n + 1
        Array1 = np.array(Array[0:n])

        for i in range(len(Array1)):
            arr = Array1[i].split(',')
            for i in range(0,nSamples):
                CPCtimes[i].append(arr[i*5]) # time in sec, starting from 1 sec. Already converted counts to #/cc with flow=1lpm
                try:
                    Concentrations[i].append(float(arr[i*5+1]))  # conc in cm-3
                except ValueError:
                    Concentrations[i].append(np.nan)
            
        for j in range(0,nSamples):
            for i in range(len(CPCtimes[j])):
                CPCtimes[j][i] = pd.to_datetime(Date+' '+CPCtimes[j][i], format="%m/%d/%y %H:%M:%S")#- pd.Timedelta(hours=1)#change to UTC time
            
    df = pd.DataFrame()
    for i in range(0,nSamples):
        df[f'Time{i+1}'] = CPCtimes[i]
        df[f'Concentration{i+1}'] = Concentrations[i]
    
    return df

def parse_opcdata(filepath):
    df = pd.read_csv(filepath, sep=';', parse_dates=[0])
    return df

def fix_timestamps(df):
    timestamps = df['TIME'].values
    arr_out = []
    prev = timestamps[0]
    arr_out.append(prev)

    for timestamp in timestamps[1::]:

        pres = timestamp
    
        if pres == prev:
            pres += pd.Timedelta(seconds=1)
        
        arr_out.append(pres)
        prev = pres

    return arr_out