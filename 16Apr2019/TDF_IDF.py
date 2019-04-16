#################################CREATING LABELENCODER FILE###################################
import pandas as pd
df=pd.read_excel('/Users/raghuram.b/Desktop/Exxon/LATEST/Final_OTMeta_With_Rules.xlsm',sheet_name='MetaData',usecols=[3,7,9,10,11], header=[0])
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
df = pd.read_csv("/Users/raghuram.b/Desktop/labelencoder.csv", encoding = "ISO-8859-1")
df["content"] = ""
del df['Unnamed: 0']
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


def clean_text(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


df['content'] = df['content'].map(lambda com : clean_text(com))



df['content'][1]



train, test = train_test_split(df, random_state=45, test_size=0.40, shuffle=True)




X_train = train.content
X_test = test.content
print(X_train.shape)
print(X_test.shape)



# Define a pipeline combining a text feature extractor with multi lable classifier
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
#


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
#
#
#logisticRegr_pipeline = Pipeline([
#                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                ('clf', OneVsRestClassifier(LogisticRegression(solver = 'lbfgs'))) ])
#
#for category in categories:
#        print('... Processing {}'.format(category))
#        filename='logisticRegr_'+category+'.pkl'
#        model=logisticRegr_pipeline.fit(X_train, train[category])
#        pickle.dump(model, open(filename, 'wb'))



############################(load from saved pickle file)##########################################
    

#Navie bayes 
###########
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
    print('Test accuracy for SVM algo category==>'+category+' is Accracy==>{}'.format(accuracy_score(test[category], prediction)))
        
  

# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)

accuracy_score(y_test,predictions)

                   
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

#from sklearn.svm import NuSVC
#    
#SVC_pipeline = Pipeline([
#          ..
#          ...........('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                ('clf', OneVsRestClassifier(NuSVC(kernel='rbf',degree=2,gamma=10), n_jobs=1)),
#            ])
#
#                
#for category in categories:
##        print('... Processing {}'.format(category))
#        model=SVC_pipeline.fit(X_train, train[category])
#        prediction = SVC_pipeline.predict(X_test)
#        print('Test accuracy for SVM algo category==>'+category+' is Accracy==>{}'.format(accuracy_score(test[category], prediction)))
#        
#        
#
#    
#same as above linear & poly
#rbf with gama (variations)
#from sklearn.svm.NuSVC
#from sklearn.svm.libsvm
#change testsize data check 20%
#gridcv



