import re
import pandas as pd
df1=df['FileName']
files=[]
for i in ['ZZ-VDSHP','ZZ-VRZZZ','ZZ-VDLAY','ZZ-QRCRI','ZZ-VDZZZ','EM-QRCRI','ZZ-VLMTO','ZZ-DRTQM','ZZ-URZZZ','ZZ-VLLMC','ZZ-VDADA','ZZ-BPMOC','ZZ-VXXER','ZZ-VLZZZ','ZZ-VCCAL']:
#for i in ['ZZ-VDSHP']:
    print(i)
    files.append(df1[df1.str.contains(i, flags=re.IGNORECASE, regex=True)].values)

print("length of files after taking the file patterns",sum([len(i) for i in files]))

cf = pd.DataFrame()
for txt in files:
    for i in txt:
        cf=cf.append({'FileName': i}, ignore_index=True)
cf.reset_index(inplace=True)        


df = pd.merge(cf, df, how = 'left', left_on = 'FileName', right_on = 'FileName') 
df.reset_index(inplace=True)  


#del df['FileName_y']
#df.rename(columns={'FileName_x':'FileName'},inplace=True)
#
#
#var=[]
#for i in ['EM-QRCRI']:
#    print(df1[df1.str.contains(i, flags=re.IGNORECASE, regex=True)].values)
#    var.append(df1[df1.str.contains(i, flags=re.IGNORECASE, regex=True)].values)
#sum([len(i) for i in var])    
#
#df.loc[df.Originator == 'FL - Fluor',['FileName','Originator']]
#df.Originator.value_counts()

#################################CREATING LABELENCODER FILE###################################
import pandas as pd
df=pd.read_excel('/Users/raghuram.b/Desktop/Exxon/LATEST/Final_OTMeta_With_Rules.xlsm',sheet_name='MetaData',usecols=[3,7,9,10,11], header=[0])
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

df['Originator']=df['Originator'].apply(remove_hyphen_characters)
le.fit(df['Originator'])
df['Originator']=le.transform(df['Originator'])

df['Discipline (Step 1)']=df['Discipline (Step 1)'].apply(remove_hyphen_characters)
le.fit(df['Discipline (Step 1)'])
df['Discipline (Step 1)']=le.transform(df['Discipline (Step 1)'])

df['Document Type (Step 2)']=df['Document Type (Step 2)'].apply(remove_hyphen_characters)
le.fit(df['Document Type (Step 2)'])
df['Document Type (Step 2)']=le.transform(df['Document Type (Step 2)'])

df['Document Subtype (Step 3)']=df['Document Subtype (Step 3)'].apply(remove_hyphen_characters)
le.fit(df['Document Subtype (Step 3)'])
df['Document Subtype (Step 3)']=le.transform(df['Document Subtype (Step 3)'])

columns=['Originator', 'Discipline (Step 1)','Document Type (Step 2)', 'Document Subtype (Step 3)']

for i in columns:
    print('The categories in',i,'are \n')
    print(df[i].value_counts())

    
df.to_csv('/Users/raghuram.b/Desktop/labelencoder.csv')

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


df_contentframe = df.drop(['FileName', 'content'], axis=1)
counts = []
categories = list(df_contentframe.columns.values)
for i in categories:
    counts.append((i, df_contentframe[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats


df = df.dropna()
df.head()
df['content'][1]



categories = ['Originator', 'Discipline (Step 1)', 'Document Type (Step 2)', 'Document Subtype (Step 3)']


from nltk.stem.porter import PorterStemmer
def clean_text(text):
    print("Before Text\n")
    print("############\n")
    print(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 \/]*','',text)
    text = re.sub('[\.]{2,}','',text)
    text = re.sub('[\-]{2,}','',text)
    text = re.sub('[\_]{2,}','',text)
    text = re.sub(r'[\s]+',' ',text)
    
    
    print("word count after lowercase,NON-Alphabetic & punctuation",sum([len(i) for i in text]))
    
    
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split(' ')]
    text = [i for i in text if not i.isdigit()]
    print("word count after removing stemming:",sum([len(i) for i in text])) 
    print("After Text\n")
    print("############\n")
    print(text)
    text=str(text)
    return text

#
#def clean_text(text):  
#    print(text)
#    tokens=text
##    tokens = text.lower()
##    tokens = re.sub('\W', ' ', tokens)
##    tokens = re.sub('\s+', ' ', tokens)
##    tokens = tokens.strip(' ')
##    tokens=str(tokens)
##    print(type(tokens))
#    
#
#    # convert to lower case and remove punctuation
#    import string
#    punctuations = list(string.punctuation)
#    #punctuations.append(".")
#    tokens = [word.lower() for word in tokens.split() if word not in punctuations] 
#    print("word count after lowercase & punctuation",sum([len(i) for i in tokens]))  
#    
#    
#    # remove remaining tokens that are not alphabetic
#    tokens = [word for word in tokens if  word.isalpha()] 
#    print("word count after removing Non alpha:",sum([len(i) for i in tokens]))     
#    
#   
#   
#    # stemming of words
#    from nltk.stem.porter import PorterStemmer
#    porter = PorterStemmer()
#    #tokens = [porter.stem(word) for word in tokens]
#    tokens = [porter.stem(word) for word in tokens ]
#    print("word count after removing stemming:",sum([len(i) for i in tokens]))    
#    return tokens

df['content'] = df['content'].apply(clean_text)





train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)




X_train = train.content
X_test = test.content
print(X_train.shape)
print(X_test.shape)



## Define a pipeline combining a text feature extractor with multi lable classifier
#NB_pipeline = Pipeline([
#                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                ('clf', OneVsRestClassifier(MultinomialNB(
#                    fit_prior=True, class_prior=None))),
#            ])
#                
#for category in categories:
#        print('... Processing {}'.format(category))
#        filename='NB_pipeline_'+category+'.pkl'
#        model=NB_pipeline.fit(X_train, train[category])
#        pickle.dump(model, open(filename, 'wb'))
#


#SVC_pipeline = Pipeline([
#                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
#            ])
  
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




#Random_pipeline = Pipeline([
#                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators = 300))) ])
#
#for category in categories:
#        print('... Processing {}'.format(category))
#        filename='Random_pipeline_'+category+'.pkl'
#        model=Random_pipeline.fit(X_train, train[category])
#        pickle.dump(model, open(filename, 'wb'))


#logisticRegr_pipeline = Pipeline([
#                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                ('clf', OneVsRestClassifier(LogisticRegression(solver = 'lbfgs'))) ])
#
#for category in categories:
#        print('... Processing {}'.format(category))
#        filename='logisticRegr_'+category+'.pkl'
#        model=logisticRegr_pipeline.fit(X_train, train[category])
#        pickle.dump(model, open(filename, 'wb'))
#


############################(load from saved pickle file)##########################################
    

##Navie bayes 
############
#for category in categories:
#    filename='NB_pipeline_'+category+'.pkl'
#    NB_from_pickle = pickle.load(open(filename, 'rb'))
#    prediction = NB_from_pickle.predict(X_test)
#    print('Test accuracy for Naive Baye category==>'+category+' is Accracy==>{}'.format(accuracy_score(test[category], prediction)))
#        
#                 

#SVC algo
##########
for category in categories:
    filename='SVC_pipeline_'+category+'.pkl'
    SVC_from_pickle = pickle.load(open(filename, 'rb'))
    prediction = SVC_from_pickle.predict(X_test)
    print('Test accuracy for SVM algo...For category==>\"'+category+'\" and  Accuracy ==>{}'.format(accuracy_score(test[category], prediction)))
        
                     
##Random Forest 
############
#
#for category in categories:
#    filename='Random_pipeline_'+category+'.pkl'
#    RandomForest_from_pickle = pickle.load(open(filename, 'rb'))
#    prediction = RandomForest_from_pickle.predict(X_test)
#    print('Test accuracy for Random Forest category==>'+category+' is Accracy==>{}'.format(accuracy_score(test[category], prediction)))
#        
#   
##Logistic Regression
################
#
#for category in categories:
#    filename='logisticRegr_'+category+'.pkl'
#    logisticRegr_from_pickle = pickle.load(open(filename, 'rb'))
#    prediction = logisticRegr_from_pickle.predict(X_test)
#    print('Test accuracy for Logistic Regression category==>'+category+' is Accracy==>{}'.format(accuracy_score(test[category], prediction)))
#           



######################################END OF CODE####################
 