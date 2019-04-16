import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/raghuram.b/Desktop/Python/AirQualityUCI/AirQualityUCI.csv', sep=';')
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

# we notice min values for all continuous variable is -200, which according to the data set info, is the tag used for missing values.
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
T= data['T'].fillna(data['T'].mean())


df = pd.DataFrame({'S1':S1, 'S2':S1, 'S3':S3, 'S4':S4, 'S5':S5,'T':T})

bins = [-2,15,30,50]
names = ['low', 'med', 'high']
df['temperature'] = pd.cut(df['T'], bins, labels=names)

cf=data.loc[:,['CO(GT)','NMHC(GT)','C6H6(GT)','NOx(GT)','NO2(GT)']]
cf['CO(GT)']=cf['CO(GT)'].fillna(data['CO(GT)'].mean())
cf['NMHC(GT)']=cf['NMHC(GT)'].fillna(data['NMHC(GT)'].mean())
cf['C6H6(GT)']=cf['C6H6(GT)'].fillna(data['C6H6(GT)'].mean())
cf['NOx(GT)']=cf['NOx(GT)'].fillna(data['NOx(GT)'].mean())
cf['NO2(GT)']=cf['NO2(GT)'].fillna(data['NO2(GT)'].mean())
df=pd.concat([df,cf],axis=1)


#df.to_csv("AirQuality_processed.csv")

####################################################################################
#All columns approach 
####################################################################################
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing, model_selection as cross_validation, neighbors, svm
import pandas as pd
#import numpy as np
from statistics import mode
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

x=df.loc[:,['S1', 'S2', 'S3', 'S4', 'S5', 'CO(GT)', 'NMHC(GT)','C6H6(GT)', 'NOx(GT)', 'NO2(GT)']]
y=df.loc[:,['temperature']]

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=20)


##SVM
clf = svm.SVC()
#Run SVM and calculate confidence
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Svm algo",confidence)


from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train, y_train)
print("LogisticRegression algo",logisticRegr.score(X_test, y_test))

####ENSEMBLE for 4 algos (using Voting) (NEED to work on it)
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model3 = neighbors.KNeighborsClassifier()
model4 = svm.SVC(random_state=1)

model1.fit(X_train,y_train)
print(' model 1 accuracy: LogisticRegression',model1.score(X_test, y_test))

model4.fit(X_train,y_train)
print(' model 4 accuracy: SVM',model4.score(X_test, y_test))


####################################################################################
#PCA approach (combining 5 columns into 1 component)
####################################################################################
from sklearn.preprocessing import StandardScaler
features = ['S1', 'S2', 'S3', 'S4', 'S5']
# Separating out the features
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
df['component1']=principalComponents[:,0]
df['component2']=principalComponents[:,1]

x=df.loc[:,['component1','component2']]
y=df.loc[:,['temperature']]

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=20)

##SVM
clf = svm.SVC()
#Run SVM and calculate confidence
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Svm algo",confidence)


from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train, y_train)
print("LogisticRegression algo",logisticRegr.score(X_test, y_test))

####ENSEMBLE for 4 algos (using Voting) (NEED to work on it)
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model3 = neighbors.KNeighborsClassifier()
model4 = svm.SVC(random_state=1)

model1.fit(X_train,y_train)
print(' model 1 accuracy: LogisticRegression',model1.score(X_test, y_test))

model4.fit(X_train,y_train)
print(' model 4 accuracy: SVM',model4.score(X_test, y_test))

