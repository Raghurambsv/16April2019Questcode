#IMPORTING THE PACKAGES
import pandas as pd
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os as os
import pickle 
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#CUSTOM FUNCTIONS DATA CLEANING
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

def clean_text(text):
    text=str(text)
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

#Import ruleset into dataframe from Quest given Excel
df=pd.read_excel('./Final_OTMeta_With_Rules.xlsm',sheet_name='MetaData',usecols=[3,7,9,10,11,15], header=[0],encoding = "utf-8")

#Remove nulls
df=df.dropna()
df.reset_index(drop=True,inplace=True)
#converting .pdf to .txt for Text Classification Approach
df['FileName']=df['FileName'].apply(remove_PDFtoTXT)



    
    
#Intialize Label encoding (Convert Alphabets to numeric for Algorithm) for all Target Variables
from sklearn import preprocessing
le_Org = preprocessing.LabelEncoder()
le_Disc = preprocessing.LabelEncoder()
le_Dtype = preprocessing.LabelEncoder()
le_DStype = preprocessing.LabelEncoder()
le_FilePath = preprocessing.LabelEncoder()


df['Originator']=df['Originator'].apply(remove_hyphen_characters)
le_Org.fit(df['Originator'])
df['Originator']=le_Org.transform(df['Originator'])


df['Discipline']=df['Discipline (Step 1)'].apply(remove_hyphen_characters)
le_Disc.fit(df['Discipline'])
df['Discipline']=le_Disc.transform(df['Discipline'])


df['Document Type']=df['Document Type (Step 2)'].apply(remove_hyphen_characters)
le_Dtype.fit(df['Document Type'])
df['Document Type']=le_Dtype.transform(df['Document Type'])


df['Document Subtype']=df['Document Subtype (Step 3)'].apply(remove_hyphen_characters)
le_DStype.fit(df['Document Subtype'])
df['Document Subtype']=le_DStype.transform(df['Document Subtype'])

le_FilePath.fit(df['FilePath'])
df['FilePath']=le_FilePath.transform(df['FilePath'])

#Creating pickle files for label encoder for each Target Variable
with open('./PickleFiles/Orginator_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Org, file)
    
with open('./PickleFiles/Discipline_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Disc, file)
    
with open('./PickleFiles/DocType_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_Dtype, file)
    
with open('./PickleFiles/DocSubtype_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_DStype, file)
    
with open('./PickleFiles/FilePath_LableEncoder.pkl', 'wb') as file:
    pickle.dump(le_FilePath, file)    


df.to_csv('./labelencoder_Document_Digitization.csv')

#Create empty series for independent variable i.e. content of text file
df["content"] = ""


#Reading the data from text files, saving the content to dataframe
for i, row in df.iterrows():
    data = None
    if os.path.isfile('./TXTs/' + df.at[i, 'FileName']) :
        with open('./TXTs/' + df.at[i, 'FileName'], 'r') as myfile:
            data=myfile.read().replace('\n', '')
            data.strip()
    content_val = data
    df.at[i,'content'] = content_val

#cleaning the text data
df['content'] = df['content'].map(lambda com : clean_text(com))


#Remove NULLs from whole dataframe
df = df.dropna()

df=df[['content','Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath']]
#List of dependent variables
categories = ['Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath']

#Spliting the data into train and test
train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)

#Extracting input data
X_train = train.content
X_test = test.content
#print(X_train.shape)
#print(X_test.shape)

#Creating the Pipeline model
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])

#Saving the model in pickle format for each category               
for category in categories:
        print('... Processing {}'.format(category))
        filename='./PickleFiles/'+category+'.pkl'
        model=SVC_pipeline.fit(X_train, train[category])
        pickle.dump(model, open(filename, 'wb'))




#Model Testing with pickled files and test data
for category in categories:
    filename='./PickleFiles/'+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
    print('Test accuracy for  category==>'+category+'  Accracy : {}'.format(accuracy_score(test[category], prediction)))
                     

############################################ END OF MODEL BUILDING AND TESTING #################################################### 
