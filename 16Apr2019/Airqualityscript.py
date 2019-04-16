import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('AirQualityUCI.csv', sep=';')
data.head()
data.info()
data = data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
#Count and display missing values.
print(data.isnull().sum())

#check all the null values where it exists
null_data = data[data.isnull().any(axis=1)]
null_data.head()

#drop all null values and its columns
data = data.dropna()
data.shape

#By looking into descriptive statistics, we notice min values for all continuous variable is -200, which according to the data set info, is the tag used for missing values.
print(data.describe())


#Replace -200 values with NaN, and inspect again with descriptive statistics.
data = data.replace(-200, np.nan)
data.describe()
#Count new missing values in data.
data.isnull().sum()

print(data.index)

data.loc[:,'Datetime'] = data['Date'] + ' ' + data['Time']

#For "Datetime", convert string values to datetime data type, and store them in list "DateTime"
from datetime import datetime
DateTime = []
for x in data['Datetime']:
    DateTime.append(datetime.strptime(x,'%d/%m/%Y %H.%M.%S'))
    
#Convert DateTime list to series, and use it as the index of data
datetime = pd.Series(DateTime)
data.index = datetime    

#By showing the first five records of dataframe, I notice some columns (e.g. CO(GT)) have multiple values per observation, which does not make much realistic sense.
data.head()

#So I checked back in the excel datasheet, and figured that cvs messed up "." with ",". The following steps are to replace "," with "." in object variables, and convert them to numerics.
data['CO(GT)'] = data['CO(GT)'].str.replace(',', '.').astype(float)
data['C6H6(GT)'] = data['C6H6(GT)'].str.replace(',','.').astype(float)
data['T'] = data['T'].str.replace(',', '.').astype(float)
data['RH'] = data['RH'].str.replace(',', '.').astype(float)
data['AH'] = data['AH'].str.replace(',', '.').astype(float)

#The last step has generated more -200 values.
data.describe()
data = data.replace(-200, np.nan)

#Identify target variables, S1-S5, and fill the missing values with the column mean
S1 = data['PT08.S1(CO)'].fillna(data['PT08.S1(CO)'].mean())
S2 = data['PT08.S2(NMHC)'].fillna(data['PT08.S1(CO)'].mean())
S3 = data['PT08.S3(NOx)'].fillna(data['PT08.S1(CO)'].mean())
S4 = data['PT08.S4(NO2)'].fillna(data['PT08.S1(CO)'].mean())
S5 = data['PT08.S5(O3)'].fillna(data['PT08.S1(CO)'].mean())

df = pd.DataFrame({'S1':S1, 'S2':S1, 'S3':S3, 'S4':S4, 'S5':S5})
df.to_csv("AirQuality_processed.csv")


fig, axes = plt.subplots(5,1, figsize=(15,24))
axes[0].plot(S1)
axes[0].set_title ('S1')
axes[1].plot(S2)
axes[1].set_title ('S2')
axes[2].plot(S3)
axes[2].set_title ('S3')
axes[3].plot(S4)
axes[3].set_title ('S4')
axes[4].plot(S5)
axes[4].set_title ('S5')


#Plots withn shorter period: '2004-10-04' - '2004-10-07', S1-S5
fig, axes = plt.subplots(5,1, figsize=(15,30))

axes[0].plot(S1['2004-10-04':'2004-10-07'])
axes[0].set_title ('S1')
axes[1].plot(S2['2004-10-04':'2004-10-07'])
axes[1].set_title ('S2')
axes[2].plot(S3['2004-10-04':'2004-10-07'])
axes[2].set_title ('S3')
axes[3].plot(S4['2004-10-04':'2004-10-07'])
axes[3].set_title ('S4')
axes[4].plot(S5['2004-10-04':'2004-10-07'])
axes[4].set_title ('S5')

#respective corr between the S1-S5 values
#S1_lagged = S1.shift()
#pd.DataFrame({'real': S1, 'lagged': S1_lagged}).corr()
#
#plt.scatter(S1, S1_lagged)
#plt.xlabel('S1')
#plt.ylabel('S1_lagged')
#
#S2_lagged = S2.shift()
#pd.DataFrame({'real': S2, 'lagged': S2_lagged}).corr()
#
#plt.scatter(S2, S2_lagged)
#plt.xlabel('S2')
#plt.ylabel('S2_lagged')
#
#S3_lagged = S3.shift()
#pd.DataFrame({'real': S3, 'lagged': S3_lagged}).corr()
#
#plt.scatter(S3, S3_lagged)
#plt.xlabel('S3')
#plt.ylabel('S3_lagged')
#
#S4_lagged = S4.shift()
#pd.DataFrame({'real': S4, 'lagged': S4_lagged}).corr()
#
#plt.scatter(S4, S4_lagged)
#plt.xlabel('S4')
#plt.ylabel('S4_lagged')
#
#S5_lagged = S5.shift()
#pd.DataFrame({'real': S5, 'lagged': S5_lagged}).corr()
#
#plt.scatter(S5, S5_lagged)
#plt.xlabel('S5')
#plt.ylabel('S5_lagged')


#correlation between all the input variables
#pd.DataFrame({'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5':S5}).corr()


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    

test_stationarity(S5)    


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    

test_stationarity(S5['2004-10-04':'2004-10-07'])    



