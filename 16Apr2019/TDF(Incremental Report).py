################################# Read ALL files from xls###################################
import pandas as pd
df=pd.read_excel('/Users/raghuram.b/Desktop/Exxon/LATEST/Final_OTMeta_With_Rules.xlsm',sheet_name='MetaData',usecols=[3,7,9,10,11], header=[0],names=['FileName', 'Originator','Discipline', 'Document Type', 'Document Subtype'])

df=df.dropna()
df.reset_index(drop=True,inplace=True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


def remove_hyphen_characters(text):
    text=str(text)
    text=text.replace("-","")
    text=text.replace("/","")
    text=text.replace(",","")
    return text

def remove_PDFtoTXT(text):
    text=str(text)
    text=text.replace("pdf","txt")
    return text



df['FileName']=df['FileName'].apply(remove_PDFtoTXT)

################################# PER FILE Creating Custom CATEGORY ###################################
import re
import pandas as pd
df1=df['FileName']

# List for categories
#list1=['ZZ-VDSHP','ZZ-VRZZZ','ZZ-VDLAY','ZZ-QRCRI','ZZ-VDZZZ','EM-QRCRI']
#
#Categories   No_of_files   
#========================                       
#ZZ-VDSHP	684
#ZZ-VRZZZ	261
#ZZ-VDLAY	209
#ZZ-QRCRI	183
#ZZ-VDZZZ	141
#EM-QRCRI	100

list1=['ZZ-VDSHP']  

files=[]
print('Files considered for this run are as below:')
for i in list1[:]:
    print(i)
    files.append(df1[df1.str.contains(i, flags=re.IGNORECASE, regex=True)].values)
print("no of files in sub-file-list",sum([len(i) for i in files]))      

def output_extract(text):
    pattern=text.split('-')
    part1=str(pattern[1:2])
    part2=str(pattern[2:3])
    part1=str(part1)
    part1=part1.replace("[","")
    part1=part1.replace("]","")
    part1=part1.replace("\'","")
    part2=str(part2)
    part2=part2.replace("[","")
    part2=part2.replace("]","")
    part2=part2.replace("\'","")
    result=part1+'-'+part2
    return str(result)

df['category']=df['FileName'].apply(output_extract)


df.loc[~(df['category'] == 'ZZ-VDSHP'), 'category'] = 'ALL'
print(df.category.value_counts())

 
print("NULL CHECKING FOR SUBFILELIST AFTER MERGE\n\n",df.isnull().sum())
print('\ndf incremental file count',len(df))

    
#################################CREATING LABELENCODER FILE###################################

df['category']=df['category'].apply(remove_hyphen_characters)
le.fit(df['category'])
df['category']=le.transform(df['category'])

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

for i in columns:
    print('The categories in',i,'are \n')
    print(df[i].value_counts())


print("NULL CHECKING FOR AFTER LABELENCODER",df.isnull().sum())

#################################TF_IDF to ACCRUACY###################################

get_ipython().run_line_magic('matplotlib', 'inline')
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
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import os as os
import pickle 
df["content"] = ""


#Selecting All TextFiles content and loading them in Dataframe

for i, row in df.iterrows():
    data = None
    if os.path.isfile('/Users/raghuram.b/Desktop/Exxon/LATEST/TXTs/' + df.at[i, 'FileName']) :
        with open('/Users/raghuram.b/Desktop/Exxon/LATEST/TXTs/' + df.at[i, 'FileName'], 'r') as myfile:
            data=myfile.read().replace('\n', '')
            data.strip()
    content_val = data
    df.at[i,'content'] = content_val
df.head()




categories = ['Originator', 'Discipline', 'Document Type', 'Document Subtype']


#######Cleaning and Preprocessing Data
from nltk.stem.porter import PorterStemmer
def clean_text(text):
    text=str(text)
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


#Calling the Clean Procedure
df['content'] = df['content'].apply(clean_text)




######PIPELINE PROCESS STARTS

from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words,lowercase=False)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])


input_variable=['content','category']
output_variable=['Originator']
#X_train,X_test,y_train,y_test = train_test_split(df[input_variable].to_xarray(),df[output_variable], random_state=42, test_size=0.33, shuffle=True)
X_train,X_test,y_train,y_test = train_test_split(df[input_variable],df[output_variable], random_state=42, test_size=0.33, shuffle=True)


X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


SVC_pipeline.fit(X_train, y_train)
prediction = SVC_piipeline.predict(X_test)
score=accuracy_score(X_test, prediction) * 100
print('printing only for Orginator (for Ex) ==> {}'.format(round(score,2)))
        


    


