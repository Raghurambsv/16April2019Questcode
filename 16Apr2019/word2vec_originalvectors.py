import gensim
import numpy as np
from gensim.models import word2vec
import re
import nltk
from time import time  # To time our operations
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#Running word2vec for all TEXT files
import glob2
#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/BIRF-EM-DDZZZ-510*")
filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/*")
totalfiles=[]
df = pd.DataFrame(columns=['filename','vector'])

###############################FILE PROCESING STARTS###########################
for FILENAME in filelist:
    print(FILENAME)       
    totalfiles.append(FILENAME.split("/")[-1])
       
    
    with open(FILENAME,'r') as file:
        content=file.readlines()
    #remove "\n" from sentences
    tokens = [w.replace('\n', '') for w in content]
    print("Initial word count from file",sum([len(i) for i in tokens]))  
    
    
    # convert to lower case and remove punctuation
    import string
    punctuations = list(string.punctuation)
    #punctuations.append(".")
    tokens = [[word for word in token.lower().split() if word not in punctuations] for token in tokens]
    print("word count after lowercase & punctuation",sum([len(i) for i in tokens]))  
    
    
    # remove remaining tokens that are not alphabetic
    tokens = [[word for word in token if  word.isalpha()] for token in tokens]
    print("word count after removing Non alpha:",sum([len(i) for i in tokens]))     
    
    
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [[word for word in token if word not in stop_words] for token in tokens]
    print("word count after removing stopwords:",sum([len(i) for i in tokens])) 
    
    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    #tokens = [porter.stem(word) for word in tokens]
    tokens = [[porter.stem(word) for word in token ] for token in tokens]
    print("word count after removing stemming:",sum([len(i) for i in tokens])) 
    
    
    # Removing empty lists if any formed
    tokens = filter(None, tokens)
    tokens=list(tokens)
    print("word count after removing empty lists",len(tokens))
    
    #Word2Vec model
    model = gensim.models.Word2Vec(tokens,workers=5,size=2,min_count=1,window=7)
   

    only_vectors = [vec for w, vec in zip(model.wv.index2word, model.wv.vectors)]
    print("After processing in gensim count:",len(model.wv.vocab) ) 

    
   
    #Adding  all words of indivual textfiles into a Single_Row in dataframe    
    df = df.append({'vector': only_vectors ,'filename': FILENAME}, ignore_index=True)


###############################FILE PROCESING ENDS###########################

#Functioon to extract the output pattern from filename
def output_extract(text):
    pattern=text.split('-')
    pattern=pattern[2:3]
    return pattern

df['pattern']=df['filename'].apply(output_extract)

#Formatting of Dataframe to get the desired Output column
def remove_extra_characters(text):
    text=str(text)
    text=text.replace("[","")
    text=text.replace("]","")
    return text
df['pattern']=df['pattern'].apply(remove_extra_characters)   #delete '[' & ']' characters
del df['filename']  #delete processed extra  column

print("Total text files from processed :",len(totalfiles))

#Converting output to numeric (LabelEncoding)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['pattern'])
df['output']=le.transform(df['pattern'])
del df['pattern'] #delete processed extra  column

##Flattening array
#df['vector']=df['vector'].apply(lambda x: np.array(x).flatten()).apply(np.array)


#Calling Algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
xtrain, xtest, ytrain, ytest = train_test_split(df['vector'].values, df['output'].values, random_state=42, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
model= RandomForestClassifier(random_state=42)
model.fit(xtrain, ytrain)
print('Random forest score:', model.score(xtest,ytest))
