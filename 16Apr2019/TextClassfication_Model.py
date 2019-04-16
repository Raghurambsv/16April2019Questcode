################################# TEXT CLASSIFICATION###################################
import pandas as pd
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os as os
import pickle 
dp_start=time.time()

################################# Read from Quest XLS###################################
df=pd.read_excel('./Otmeta.xlsx',sheet_name='MetaData',usecols=[0,7,9,10,11], header=[0],names=['FileName', 'Originator','Discipline', 'DocType', 'DocSubType'])
part=''
df=df.dropna()
df.reset_index(inplace=True)




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

def extract_doctype(text):
#    text=str(text.encode("utf-8"))
    text=text[0]
    return text

def extract_docsubtype(text):
#    text=str(text.encode("utf-8"))
    text=text[0:3]
    return text

#To keep only 2 categories in Orginator columns
def Originator_Correction(text):
    text=str(text)
    text=text.replace('FL - Fluor','ZZ - Not a Primary Originator')
    text=text.replace('KB - KBR (Kellogg Brown & Root LLC)','ZZ - Not a Primary Originator')
    text=text.replace('AH - Anvil Engineering Home Office','ZZ - Not a Primary Originator')
    return text
    


df['FileName']=df['FileName'].apply(remove_PDFtoTXT)
df['DocType']=df['DocType'].apply(extract_doctype)
df['DocSubType']=df['DocSubType'].apply(extract_docsubtype)
df['combi']=df['DocType']+df['DocSubType']
df['Originator']=df['Originator'].apply(Originator_Correction)
df['Originator_ref']=df['Originator']
df['Discipline_ref']=df['Discipline']
df['combi_ref']=df['combi']




################################# Load contents from Text file ###################################

df1=df['FileName']
files=[]
files.append(df1[df1.str.len() > 0].values)
print("No of files :",sum([len(i) for i in files]))      

#Loading Text Files content
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


#Cleaning Text files content
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

dp_end=time.time()
print('\nData preparation Time :',round(dp_end-dp_start))


################################# Originator ###################################
de_start=time.time()

#Originator Categories	
#ZZ	3510
#EM	427

columns=['Originator']


olist1=['EM - ExxonMobil','ZZ - Not a Primary Originator']

df['Originator'] = df.apply( lambda row: len(olist1) + 1, axis=1)


for i in olist1[:]:
    df['Originator'] = df.apply( lambda row: int(olist1.index(row['Originator_ref'])) if (i == row['Originator_ref']) else row['Originator'], axis=1)


#train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)
train, test = train_test_split(df, random_state=42, test_size=0.01, shuffle=True)
X_train = train.content
X_test = test.content

from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])
        

categories = ['Originator']
#Saving the Originator model in pickle format           
for category in categories:
        print('\n... Processing {} Accruacy'.format(category))
        filename='./PickleFiles/'+part+category+'.pkl'
        model=SVC_pipeline.fit(X_train, train[category])
        pickle.dump(model, open(filename, 'wb'))
       

#Originator Model Testing with pickled files and displaying Accuracy
for category in categories:
    filename='./PickleFiles/'+part+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
    print('Test accuracy for  category==>'+category+'  Accuracy : {}'.format(accuracy_score(test[category], prediction)))
         

################################# Discipline ###################################
    
#Discipline Categories
#V - Vendor/Supplier/Manufacturer	2192
#Q - QA/QC, Quality Management	512
#D - Fixed Equipment	268
#P - Process	187
#M - Rotating Equipment	176
#L - Piping	145
#B - Business & General	128
#U - Fired Equipment	114
#others - 215
    
def Discipline_Correction(text):
    text=str(text)
    text=text.replace('E - Electrical','others')
    text=text.replace('R - Materials, Corrosion, Material Technology','others')
    text=text.replace('O - Operations','others')
    text=text.replace('X - Procurement','others')
    text=text.replace('K - Construction','others')
    text=text.replace('S - Safety, Health, and Environment (SHE), Regulatory, Security','others')
    text=text.replace('N - Structural','others')
    text=text.replace('F - Risk & Loss Prevention','others')
    text=text.replace('I - Instrumentation, Metering and Process Control','others')
    text=text.replace('W - Systems Completions, Commissioning, & Start-Up','others')
    text=text.replace('C - Civil & Architectural','others')
    return text

df['Discipline']=df['Discipline'].apply(Discipline_Correction)
df['Discipline_ref']=df['Discipline_ref'].apply(Discipline_Correction)
    
columns=['Discipline']



dlist1=['B - Business & General','D - Fixed Equipment','L - Piping','M - Rotating Equipment','P - Process','Q - QA/QC, Quality Management','U - Fired Equipment','V - Vendor/Supplier/Manufacturer','others']

df['Discipline'] = df.apply( lambda row: len(dlist1) + 1, axis=1)

index = 0
for i in dlist1[:]:
    df['Discipline'] = df.apply( lambda row: int(dlist1.index(row['Discipline_ref'])) if (i == row['Discipline_ref']) else row['Discipline'], axis=1)
    index = index + 1
    


#train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)
train, test = train_test_split(df, random_state=42, test_size=0.01, shuffle=True)
X_train = train.content
X_test = test.content

from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])
        

categories = ['Discipline']
#Saving the Discipline model in pickle format for each category              
for category in categories:
        print('\n... Processing {} Accruacy'.format(category))
        filename='./PickleFiles/'+part+category+'.pkl'
        model=SVC_pipeline.fit(X_train, train[category])
        pickle.dump(model, open(filename, 'wb'))
       

#Discipline Model Testing with pickled files and displaying Accuracy
for category in categories:
    filename='./PickleFiles/'+part+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
    print('Test accuracy for  category==>'+category+'  Accuracy : {}'.format(accuracy_score(test[category], prediction)))
         


################################# DocType & DocSubType ###################################
    
#DocumentType & Subtype Categories	
#DSHP	760
#RZZZ	327
#RCRI	292
#DLAY	232
#DISO	123
#DPID	114
#XXER	111
#RTQM	93
#Others	1885

columns=['combi']

#To keep only above 8 categories and rest "others" in DocType & DocSubType combined columns
def DTST_Correction(text):
    text=str(text)
    text=text.replace('LMTO','others')
    text=text.replace('DADA','others')
    text=text.replace('PMOC','others')
    text=text.replace('DSLD','others')
    text=text.replace('LZZZ','others')
    text=text.replace('MZZZ','others')
    text=text.replace('TTDS','others')
    text=text.replace('DZZZ','others')
    text=text.replace('LLMC','others')
    text=text.replace('CZZZ','others')
    text=text.replace('RTST','others')
    text=text.replace('RAUD','others')
    text=text.replace('XXEP','others')
    text=text.replace('RMIL','others')
    text=text.replace('DDET','others')
    text=text.replace('DEQL','others')
    text=text.replace('RCAR','others')
    text=text.replace('QMAT','others')
    text=text.replace('DPFD','others')
    text=text.replace('MVMC','others')
    text=text.replace('SZZZ','others')
    text=text.replace('DMEC','others')
    text=text.replace('RSRV','others')
    text=text.replace('PWLD','others')
    text=text.replace('SSTD','others')
    text=text.replace('RCOR','others')
    text=text.replace('CMEC','others')
    text=text.replace('BRUB','others')
    text=text.replace('EZZZ','others')
    text=text.replace('MOPR','others')
    text=text.replace('SPDS','others')
    text=text.replace('QRAD','others')
    text=text.replace('RRSK','others')
    text=text.replace('XTBE','others')
    text=text.replace('ECUR','others')
    text=text.replace('RVRN','others')
    text=text.replace('BDES','others')
    text=text.replace('DSCM','others')
    text=text.replace('PZZZ','others')
    text=text.replace('YRFQ','others')
    text=text.replace('EMEC','others')
    text=text.replace('RCQS','others')
    text=text.replace('LMEL','others')
    text=text.replace('PITP','others')
    text=text.replace('LITP','others')
    text=text.replace('DCEM','others')
    text=text.replace('PERE','others')
    text=text.replace('RMCM','others')
    text=text.replace('QMRB','others')
    text=text.replace('XSOW','others')
    text=text.replace('DPED','others')
    text=text.replace('DSTR','others')
    text=text.replace('YECS','others')
    text=text.replace('XBUL','others')
    text=text.replace('XXCP','others')
    text=text.replace('RVES','others')
    text=text.replace('MMNT','others')
    text=text.replace('RITP','others')
    text=text.replace('YZZZ','others')
    text=text.replace('MSCH','others')
    text=text.replace('QZZZ','others')
    text=text.replace('YRFP','others')
    text=text.replace('RBFD','others')
    text=text.replace('YCON','others')
    text=text.replace('BSOW','others')
    text=text.replace('QWLD','others')
    text=text.replace('RSEC','others')
    text=text.replace('BZZZ','others')
    text=text.replace('EVES','others')
    text=text.replace('QPQR','others')
    text=text.replace('QMIL','others')
    text=text.replace('QRTR','others')
    text=text.replace('EELE','others')
    text=text.replace('RDEF','others')
    text=text.replace('RHYD','others')
    text=text.replace('DPEL','others')
    text=text.replace('YBUD','others')
    text=text.replace('QULT','others')
    text=text.replace('RMAT','others')
    text=text.replace('PPUR','others')
    text=text.replace('DBLK','others')
    text=text.replace('BPDC','others')
    text=text.replace('YAFE','others')
    text=text.replace('ROPR','others')
    text=text.replace('GZZZ','others')
    text=text.replace('XXSU','others')
    text=text.replace('DANC','others')
    text=text.replace('RNCR','others')
    text=text.replace('RMET','others')
    text=text.replace('RWLD','others')
    text=text.replace('PPPP','others')
    text=text.replace('MLUB','others')
    text=text.replace('PCEP','others')
    text=text.replace('DSYM','others')
    text=text.replace('QPQC','others')
    text=text.replace('RNDE','others')
    text=text.replace('RFRB','others')
    text=text.replace('DAED','others')
    text=text.replace('RCCP','others')
    text=text.replace('CRCC','others')
    text=text.replace('CHMB','others')
    text=text.replace('RSHE','others')
    text=text.replace('LBLT','others')
    text=text.replace('PPTW','others')
    text=text.replace('RPUN','others')
    text=text.replace('DSCD','others')
    text=text.replace('GEXC','others')
    text=text.replace('QCOD','others')
    text=text.replace('QSCC','others')
    text=text.replace('MMSP','others')
    text=text.replace('YSCH','others')
    text=text.replace('DCPT','others')
    text=text.replace('CPIP','others')
    text=text.replace('REAS','others')
    text=text.replace('DWDS','others')
    text=text.replace('DDPT','others')
    text=text.replace('RPPL','others')
    text=text.replace('PFAT','others')
    text=text.replace('MPCP','others')
    text=text.replace('PENG','others')
    text=text.replace('QSCN','others')
    text=text.replace('RFRC','others')
    text=text.replace('TDPT','others')
    text=text.replace('YEST','others')
    text=text.replace('CPSA','others')
    text=text.replace('EPIP','others')
    text=text.replace('CPUC','others')
    text=text.replace('CPDC','others')
    text=text.replace('CPRC','others')
    text=text.replace('MTRN','others')
    text=text.replace('PWRK','others')
    text=text.replace('DFND','others')
    text=text.replace('QCOR','others')
    text=text.replace('PCRR','others')
    text=text.replace('DPSD','others')
    text=text.replace('PENV','others')
    text=text.replace('RPQS','others')
    text=text.replace('YCLD','others')
    text=text.replace('REST','others')
    text=text.replace('REXE','others')
    return text

df['combi']=df['combi'].apply(DTST_Correction)
df['combi_ref']=df['combi_ref'].apply(DTST_Correction)


tlist1=['DISO', 'DLAY', 'DPID', 'DSHP', 'RCRI', 'RTQM', 'RZZZ', 'XXER','others']

df['combi'] = df.apply( lambda row: len(tlist1) + 1, axis=1)

for i in tlist1[:]:
    df['combi'] = df.apply( lambda row: int(tlist1.index(row['combi_ref'])) if (i == row['combi_ref']) else row['combi'], axis=1)

#train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)
train, test = train_test_split(df, random_state=42, test_size=0.01, shuffle=True)
X_train = train.content
X_test = test.content

from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])
        
categories = ['combi']
#Saving the DocType & DocSubType model in pickle format for each category             
for category in categories:
        print('\n... Processing {} Accruacy'.format(category))
        filename='./PickleFiles/'+part+category+'.pkl'
        model=SVC_pipeline.fit(X_train, train[category])
        pickle.dump(model, open(filename, 'wb'))
       

#DocType & DocSubType Model Testing with pickled files and displaying accuracy
for category in categories:
    filename='./PickleFiles/'+part+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
    print('Test accuracy for  category==>'+category+'  Accuracy : {}'.format(accuracy_score(test[category], prediction)))
         


                    
#Total Executoin time for program 
de_end=time.time()
print('\nModel Execution Time',round(de_end-de_start))    




################################# Pickle File preparation ###################################
from sklearn import preprocessing
le_Org = preprocessing.LabelEncoder()
le_Disc = preprocessing.LabelEncoder()
le_DTSTtype = preprocessing.LabelEncoder()

tlist1=['DISO', 'DLAY', 'DPID', 'DSHP', 'RCRI', 'RTQM', 'RZZZ', 'XXER','others']
dlist1=['B', 'D', 'L', 'M', 'P', 'Q', 'U', 'V', 'others']
olist1=['EM','ZZ']


le_Org.fit(np.array(['EM','ZZ']))
le_Org.transform(np.array(['EM','ZZ']))

le_Disc.fit(np.array(['B', 'D', 'L', 'M', 'P', 'Q', 'U', 'V', 'others']))
le_Disc.transform(np.array(['B', 'D', 'L', 'M', 'P', 'Q', 'U', 'V', 'others']))

le_DTSTtype.fit(np.array(['DISO', 'DLAY', 'DPID', 'DSHP', 'RCRI', 'RTQM', 'RZZZ', 'XXER','others']))
le_DTSTtype.transform(np.array(['DISO', 'DLAY', 'DPID', 'DSHP', 'RCRI', 'RTQM', 'RZZZ', 'XXER','others']))



df.to_csv('./labelencoder_Document_Digitization.csv')

#Creating pickle files for label encoder for each Target Variable
with open('./PickleFiles/labelencoder/'+part+'Orginator_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Org, file)
    
with open('./PickleFiles/labelencoder/'+part+'Discipline_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Disc, file)
    
with open('./PickleFiles/labelencoder/'+part+'Type_and_Subtype_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_DTSTtype, file)



######################################END OF CODE####################
