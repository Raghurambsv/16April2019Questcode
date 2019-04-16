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
################################# PER FILE CATEGORY ###################################
import re
import pandas as pd
df1=df['FileName']


#list1=['ZZ-VDSHP','ZZ-VRZZZ','ZZ-VDLAY','ZZ-QRCRI','ZZ-VDZZZ','EM-QRCRI','ZZ-VLMTO','ZZ-DRTQM','ZZ-URZZZ','ZZ-VLLMC','ZZ-VDADA','ZZ-BPMOC','ZZ-VXXER','ZZ-VLZZZ','ZZ-VCCAL','ZZ-VRMIL','ZZ-QRZZZ','ZZ-VMZZZ','ZZ-VXXEP','ZZ-VQMAT','ZZ-QRAUD','ZZ-VMVMC','ZZ-VPWLD','ZZ-VDDET','EM-DDZZZ']

#list1=[,,,,'',,'',,,,,,,,,,,,,,,,,,]
list1=['ZZ-VDSHP','ZZ-VRZZZ','ZZ-VDLAY','ZZ-QRCRI','ZZ-VDZZZ',
       'EM-QRCRI','ZZ-VLMTO','ZZ-DRTQM','ZZ-URZZZ','ZZ-VLLMC',
       'ZZ-VDADA','ZZ-BPMOC','ZZ-VXXER','ZZ-VLZZZ','ZZ-VCCAL',
       'ZZ-VRMIL', 'ZZ-QRZZZ', 'ZZ-VMZZZ', 'ZZ-VXXEP','ZZ-VQMAT','ZZ-QRAUD','ZZ-VMVMC','ZZ-VPWLD','ZZ-VDDET','EM-DDZZZ']
#for num in range(1,25):


files=[]
print('Files considered for this run are as below:')
for i in list1[:]:
    print(i)
    files.append(df1[df1.str.contains(i, flags=re.IGNORECASE, regex=True)].values)
print("no of files in sub-file-list",sum([len(i) for i in files]))      

cf=pd.DataFrame(columns=['FileName'])
for txt in files:
    for i in txt:
        cf=cf.append({'FileName': i}, ignore_index=True)
#    cf.reset_index(inplace=True)        
df = pd.merge(cf, df, how = 'left', left_on = 'FileName', right_on = 'FileName') 
#    df.reset_index(inplace=True) 

print("NULL CHECKING FOR SUBFILELIST AFTER MERGE\n\n",df.isnull().sum())
print('\ndf incremental file count',len(df))

    



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

for i in columns:
    print('The categories in',i,'are \n')
    print(df[i].value_counts())


#print('For files in ==>',list1[:count],'Originator value_counts\n\n',df['Originator'].value_counts()) 
#print('For files in ==>',list1[:count],'Discipline value_counts\n\n',df['Discipline (Step 1)'].value_counts())
#print('For files in ==>',list1[:count],'DocType value_counts\n\n',df['Document Type (Step 2)'].value_counts())
#print('For files in ==>',list1[:count],'SubType value_counts\n\n',df['Document Subtype (Step 3)'].value_counts())
#   

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
#df = pd.read_csv("/Users/raghuram.b/Desktop/labelencoder.csv", encoding = "ISO-8859-1")
df["content"] = ""
#del df['Unnamed: 0']
df.head()



# In[20]:


for i, row in df.iterrows():
    data = None
    if os.path.isfile('/Users/raghuram.b/Desktop/Exxon/LATEST/TXTs/' + df.at[i, 'FileName']) :
        with open('/Users/raghuram.b/Desktop/Exxon/LATEST/TXTs/' + df.at[i, 'FileName'], 'r') as myfile:
            data=myfile.read().replace('\n', '')
            data.strip()
    content_val = data
    df.at[i,'content'] = content_val
df.head()


# In[21]:


#df_contentframe = df.drop(['FileName', 'content'], axis=1)
#counts = []
#categories = list(df_contentframe.columns.values)
#for i in categories:
#    counts.append((i, df_contentframe[i].sum()))
#df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
#df_stats
#
#
#df = df.dropna()
#df.head()
#df['content'][1]



categories = ['Originator', 'Discipline', 'Document Type', 'Document Subtype']


from nltk.stem.porter import PorterStemmer
def clean_text(text):
#    print("Before Text\n")
#    print("############\n")
#    print(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 \/]*','',text)
    text = re.sub('[\.]{2,}','',text)
    text = re.sub('[\-]{2,}','',text)
    text = re.sub('[\_]{2,}','',text)
    text = re.sub(r'[\s]+',' ',text)
      
#    print("word count after lowercase,NON-Alphabetic & punctuation",sum([len(i) for i in text]))
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split(' ')]
    text = [i for i in text if not i.isdigit()]
#    print("word count after removing stemming:",sum([len(i) for i in text])) 
#    print("After Text\n")
#    print("############\n")
#    print(text)
    text=str(text)
    return text


df['content'] = df['content'].apply(clean_text)





train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)




X_train = train.content
X_test = test.content
#print(X_train.shape)
#print(X_test.shape)



from sklearn.svm import SVC      
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=1)),
            ])
        

                
for category in categories:
        print('... Processing {}'.format(category))
        filename='SVC_pipeline_'+category+'.pkl'
        model=SVC_pipeline.fit(X_train, train[category])
        pickle.dump(model, open(filename, 'wb'))



#SVC algo
##########
for category in categories:
    filename='SVC_pipeline_'+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
    score=accuracy_score(test[category], prediction) * 100
    print(category,' ==> {}'.format(round(score,2)))
        
print("no of files in sub-file-list",sum([len(i) for i in files])) 

######################################END OF CODE####################
 