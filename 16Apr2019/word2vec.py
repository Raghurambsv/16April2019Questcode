import gensim
from gensim.models import word2vec
#import logging  # Setting up the loggings to monitor gensim
#logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
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
for FILENAME in filelist:
    print(FILENAME)       
    totalfiles.append(FILENAME.split("/")[-1])
       
    
    #with open('/Users/raghuram.b/Desktop/Exxon/TXTs/testing.txt','r') as file:
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
    
    #
    ## filter out stop words
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
    
    
    model = gensim.models.Word2Vec(tokens,workers=9,size=150,min_count=1,window=7)
    #model.train(tokens, total_examples=model.corpus_count, epochs=30,report_delay=1)
    #t = time()
    #print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    #print(model)
    #words = list(model.wv.vocab)
    #print("The vocab of textfile learnt",words)
    #print("After processing in gensim count:",len(words) ) 
    #print(model.corpus_total_words,"words are reduced to",len(words),"thru word2vec","&  Shape of file is ",model.wv.vectors.shape)
    
    
    #print("each indivual <word> ka <number> format <give input as word>",model.wv['movement'])
    #print(model.wv.most_similar('provide'))
    #print(model.wv.similarity('provide', 'gauge'))
    
    
    # To make the model memory efficient
    model.init_sims(replace=True)                              
    
    # Saving the model for later use. Can be loaded using Word2Vec.load()
    FILENAME=FILENAME.split("/")[-1]
    FILENAME_WORD2VEC=FILENAME.split('.')[0]
    FILENAME_WORD2VEC='/Users/raghuram.b/Desktop/Exxon/Generated_word2vec/'+FILENAME_WORD2VEC
    model.save(FILENAME_WORD2VEC)
    #or
    #model.wv.save_word2vec_format("300features_40minwords_10context", binary=True)

    
    

print("Total text files from processed :",len(totalfiles))




#Direct from WORD2VEC folder only extracting files

import glob2
import pandas as pd
#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/BIRF-EM-DDZZZ-510*")
filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/Generated_word2vec/*")
totalfiles=[]
for FILENAME in filelist:
    print(FILENAME)       
    totalfiles.append(FILENAME.split("/")[-1])

def output_extract(text):
    pattern=text.split('-')
    pattern=pattern[2:3]
    return pattern
#put things in Dataframe    
df=pd.DataFrame(totalfiles,columns=['filename'])    
df['pattern']=df['filename'].apply(output_extract)  
print(df)







################################################ END OF CODE #########################################
#For loading the saved file
#from gensim.test.utils import datapath
#from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
#word_vectors = KeyedVectors.load('300features_40minwords_10context', mmap='r')
#word_vectors=Word2Vec.load('300features_40minwords_10context')

#vocab = list(model.wv.vocab.keys())
#print(vocab[:10])
#
#list=model.wv['b']
#print(len(list))
#model.save(model_name)
#model.train_count
#model.corpus_total_words
#model.wv.index2word
#model.wv.vocab.keys()
#https://nlpforhackers.io/word-embeddings/ --word2vec graph presentation



#Read data from xls and get the output
#import csv
#import xlrd
#import pandas as pd
#import numpy as np
#output=[]
#filename=[]
#workbook = xlrd.open_workbook('/Users/raghuram.b/Desktop/Exxon/Final_OTMeta_With_Rules.xlsm')
#sheet = workbook.sheet_by_index(1)
#for i in range(sheet.nrows):
##    print(sheet.row_slice(rowx=i,start_colx=3,end_colx=4))
#    temp=str(sheet.row_slice(rowx=i,start_colx=3,end_colx=4))
#    output.append(temp)
#    
#
#
#label=[]
#for i in range(0,len(output)):
#    pattern=output[i].split('-')
#    pattern=pattern[2:3]
#    label.append(pattern)





  


    