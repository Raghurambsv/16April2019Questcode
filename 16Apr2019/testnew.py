################################# Read ALL files from xls###################################
import pandas as pd
import time
dp_start=time.time()
df=pd.read_excel('./Final_OTMeta_With_Rules.xlsm',sheet_name='MetaData',usecols=[3,7,9,10,11], header=[0],names=['FileName', 'Originator','Discipline', 'Document Type', 'Document Subtype'])

df=df.dropna()
df.reset_index(drop=True,inplace=True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


def remove_hyphen_characters(text):
    text=str(text.encode("utf-8"))
    text=text.replace("-","")
    text=text.replace("/","")
    text=text.replace(",","")
    return text

def remove_PDFtoTXT(text):
    text=str(text)
    text=text.replace("pdf","txt")
    return text



df['FileName']=df['FileName'].apply(remove_PDFtoTXT)
################################# PER FILE CATEGORY ###################################
import re
import pandas as pd
df1=df['FileName']

files=[]
#print('Files considered for this run are as below:')
files.append(df1[df1.str.len() > 0].values)
#print("No of files in sub-file-list",sum([len(i) for i in files]))   
print("No of files :",sum([len(i) for i in files]))      

cf=pd.DataFrame(columns=['FileName'])
for txt in files:
    for i in txt:
        cf=cf.append({'FileName': i}, ignore_index=True)
df = pd.merge(cf, df, how = 'left', left_on = 'FileName', right_on = 'FileName') 

#print("NULL CHECKING FOR SUBFILELIST AFTER MERGE\n\n",df.isnull().sum())
#print('\ndf incremental file count',len(df))

#################################CREATING LABELENCODER FILE###################################

df['Originator']=df['Originator'].apply(remove_hyphen_characters)
le.fit(df['Originator'])
df['Originator']=le.transform(df['Originator'])

df['Discipline']=df['Discipline'].apply(remove_hyphen_characters)
le.fit(df['Discipline'])
df['Discipline']=le.transform(df['Discipline'])

df['Document Type']=df['Document Type'].apply(remove_hyphen_characters)
le.fit(df['Document Type'])
df['Document Type']=le.transform(df['Document Type'])

df['Document Subtype']=df['Document Subtype'].apply(remove_hyphen_characters)
le.fit(df['Document Subtype'])
df['Document Subtype']=le.transform(df['Document Subtype'])

columns=['Originator', 'Discipline', 'Document Type', 'Document Subtype']

# for i in columns:
#     print('The categories in',i,'are \n')
#     print(df[i].value_counts())

df.to_csv('./labelencoder.csv')


#Categories   No_of_files   
#========================                       
#ZZ-VDSHP	        684
#ZZ-VRZZZ         	261
#ZZ-VDLAY	        209
#ZZ-QRCRI	        183
#ZZ-VDZZZ	        141
#EM-QRCRI	        100

#list1=['ZZ-VDSHP', 'ZZ-VRZZZ', 'ZZ-VDLAY', 'ZZ-QRCRI', 'ZZ-VDZZZ', 'EM-QRCRI', 'ZZ-VLMTO']

#list1=['ZZ-VDSHP','ZZ-VRZZZ','ZZ-VDLAY','ZZ-QRCRI','ZZ-VDZZZ','EM-QRCRI','ZZ-VLMTO']
#olist1=['ZZ','EM']
#dlist1=['V','Q']
#dtlist1=['D','R','L']
#dstlist1=['SHP','ZZZ','LAY','CRI','MTO']

list1=['ZZ-VDSHP','ZZ-VRZZZ']
olist1=['ZZ']
dlist1=['V']
dtlist1=['D','R']
dstlist1=['SHP','ZZZ']



df['Originator'] = df.apply( lambda row: len(list1) + 1, axis=1)
df['Discipline'] = df.apply( lambda row: len(list1) + 1, axis=1)
df['Document Type'] = df.apply( lambda row: len(list1) + 1, axis=1)
df['Document Subtype'] = df.apply( lambda row: len(list1) + 1, axis=1)

#print("NULL CHECKING FOR AFTER LABELENCODER\n",df.isnull().sum())

#################################TF_IDF to ACCRUACY###################################

import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import os as os
import pickle 
df["content"] = ""
df.head()

for i, row in df.iterrows():
    data = None
    if os.path.isfile('./TXTs/' + df.at[i, 'FileName']) :
        with open('./TXTs/' + df.at[i, 'FileName'], 'r') as myfile:
            data=myfile.read().replace('\n', '')
            data.strip()
    content_val = data
    df.at[i,'content'] = content_val
df.head()

df = df.dropna()

categories = ['Originator', 'Discipline', 'Document Type', 'Document Subtype']

from nltk.stem.porter import PorterStemmer
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 \/]*','',text)
    text = re.sub('[\.]{2,}','',text)
    text = re.sub('[\-]{2,}','',text)
    text = re.sub('[\_]{2,}','',text)
    text = re.sub(r'[\s]+',' ',text)
      
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split(' ')]
    text = [i for i in text if not i.isdigit()]
    text=str(text)
    return text

# df['content'] = df['content'].apply(clean_text)

index = 0
for i in list1[:]:
	df['Originator'] = df.apply( lambda row: olist1.index((row['FileName'])[5:7]) if (i[0:2] == (row['FileName'])[5:7]) else row['Originator'], axis=1)
	df['Discipline'] = df.apply( lambda row: dlist1.index((row['FileName'])[8:9]) if (i[3:4] == (row['FileName'])[8:9]) else row['Discipline'], axis=1)
	df['Document Type'] = df.apply( lambda row: dtlist1.index((row['FileName'])[9:10]) if (i[4:5] == (row['FileName'])[9:10]) else row['Document Type'], axis=1)
	df['Document Subtype'] = df.apply( lambda row: dstlist1.index((row['FileName'])[10:13]) if (i[5:8] == (row['FileName'])[10:13]) else row['Document Subtype'], axis=1)
	index = index + 1
dp_end=time.time()
print('\nData preparation Time :',round(dp_end-dp_start))


de_start=time.time()
train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)
X_train = train.content
X_test = test.content

from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])
        

#SVC algo
##########                
for category in categories:
        print('\n... Processing {} Accruacy'.format(category))
        SVC_pipeline.fit(X_train, train[category])
        prediction = SVC_pipeline.predict(X_test)
        score=accuracy_score(test[category], prediction) * 100
        print(category,' ==> {}'.format(round(score,2)))
        
#print("no of files in sub-file-list",sum([len(i) for i in files]))   
de_end=time.time()
print('\nData Execution Time',round(de_end-de_start))    

######################################END OF CODE####################


