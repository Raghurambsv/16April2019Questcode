################################# Read ALL files from xls###################################
import pandas as pd
import time
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
dp_start=time.time()
df=pd.read_excel('./Final_OTMeta_With_Rules.xlsm',sheet_name='MetaData',usecols=[3,7,9,10,11,15], header=[0],names=['FileName', 'Originator','Discipline', 'Document Type', 'Document Subtype','FilePath'])
part='Final_Model_'
df=df.dropna()
df.reset_index(drop=True,inplace=True)




def remove_hyphen_characters(text):
#    text=str(text.encode("utf-8"))
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

#Intialize Label encoding (Convert Alphabets to numeric for Algorithm) for all Target Variables
from sklearn import preprocessing
le_Org = preprocessing.LabelEncoder()
le_Disc = preprocessing.LabelEncoder()
le_Dtype = preprocessing.LabelEncoder()
le_DStype = preprocessing.LabelEncoder()
le_FilePath = preprocessing.LabelEncoder()

le_FilePath.fit(df['FilePath'])
df['FilePath']=le_FilePath.transform(df['FilePath'])



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



columns=['Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath']




#Categories   No_of_files   
#========================                       
#ZZ-VDSHP        684
#ZZ-VRZZZ        261
#ZZ-VDLAY        209
#ZZ-QRCRI        183
#ZZ-VDZZZ         141 
#EM-QRCRI        100 



list1=['ZZ-VDSHP', 'ZZ-VRZZZ', 'ZZ-VDLAY','ZZ-QRCRI', 'EM-QRCRI','ZZ-VDZZZ']
olist1=['EM','ZZ']
dlist1=['Q','V']
dtlist1=['D','L','R']
dstlist1=['CRI','LAY','SHP','ZZZ']





df['Originator'] = df.apply( lambda row: len(list1) + 1, axis=1)
df['Discipline'] = df.apply( lambda row: len(list1) + 1, axis=1)
df['Document Type'] = df.apply( lambda row: len(list1) + 1, axis=1)
df['Document Subtype'] = df.apply( lambda row: len(list1) + 1, axis=1)




#################################TF_IDF to ACCRUACY###################################


df["content"] = ""
df.head()

for i, row in df.iterrows():
    data = None
    if os.path.isfile('./TXTs/' + df.at[i, 'FileName']) :
        with open('./TXTs/' + df.at[i, 'FileName'], 'r',encoding="utf-8") as myfile:
            data=myfile.read().replace('\n', '')
            data.strip()
    content_val = data
    df.at[i,'content'] = content_val
df.head()



categories = ['Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath']


def clean_text(text):
    text=str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 \/]*','',text)
    text = re.sub('[\.]{2,}','',text)
    text = re.sub('[\-]{2,}','',text)
    text = re.sub('[\_]{2,}','',text)
    text = re.sub(r'[\s]+',' ',text)
    text = [ word for word in text.split(' ') if not len(word) == 1]
    text=str(text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
#    print(text)
    return text

#Calling the Clean Procedure
df['content'] = df['content'].apply(clean_text)
df['content'].dropna(inplace=True)

df = df.dropna()

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
train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)
X_train = train.content
X_test = test.content

from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])
        
categories = ['Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath']
#Saving the model in pickle format for each category              
for category in categories:
        print('\n... Processing {} Accruacy'.format(category))
        filename='./PickleFiles/'+part+category+'.pkl'
        model=SVC_pipeline.fit(X_train, train[category])
        pickle.dump(model, open(filename, 'wb'))
        
          
        

#Model Testing with pickled files and test data
for category in categories:
    filename='./PickleFiles/'+part+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
#    print(prediction)
    print('Test accuracy for  category==>'+category+'  Accracy : {}'.format(accuracy_score(test[category], prediction)))
                             
 
de_end=time.time()
print('\nData Execution Time',round(de_end-de_start))    

######################################END OF CODE####################


olist1=['EM','ZZ']
dlist1=['Q','V']
dtlist1=['D','L','R']
dstlist1=['CRI','LAY','SHP','ZZZ']


le_Org.fit(np.array(['EM','ZZ']))
le_Org.transform(np.array(['EM','ZZ']))

le_Disc.fit(np.array(['Q','V']))
le_Disc.transform(np.array(['Q','V']))

le_Dtype.fit(np.array(['D','L','R']))
le_Dtype.transform(np.array(['D','L','R']))

le_DStype.fit(np.array(['CRI','LAY','SHP','ZZZ']))
le_DStype.transform(np.array(['CRI','LAY','SHP','ZZZ']))



df.to_csv('/Users/raghuram.b/Desktop/labelencoder_Document_Digitization.csv')

#Creating pickle files for label encoder for each Target Variable
with open('./PickleFiles/labelencoder/'+part+'Orginator_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Org, file)
    
with open('./PickleFiles/labelencoder/'+part+'Discipline_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Disc, file)
    
with open('./PickleFiles/labelencoder/'+part+'DocType_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Dtype, file)
    
with open('./PickleFiles/labelencoder/'+part+'DocSubtype_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_DStype, file)
    
with open('./PickleFiles/labelencoder/'+part+'FilePath_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_FilePath, file)    
    

columns=['Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath']
for i in columns:
    print('The categories in',i,'are \n')
    print(df[i].value_counts())


