import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

import glob2
from datetime import datetime 
filename=glob2.glob("/Users/raghuram.b/Desktop/Document/images/*")

print(filename)

for path in filename:
    with open(path,'r') as file:
        line=file.readlines()

#######
#Remove punctuation and other things before GOING FURTHER
#######
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

##With defined vocabulary to filter based on our need 
myvocabulary = ['life', 'learning']
corpus = {1: "d ccc The game of life is  egg rst  game of everlasting LEARNING", 2: "The unexamined 888 life is not worth living", 3: "Never stop learning"}
#corpus = ["a b c ccc b5 The game of life is  ab cd egg rst  game of everlasting LEARNING","The unexamined 888  99 life is not worth living","Never stop learning"]
tfidf = TfidfVectorizer(vocabulary = myvocabulary, ngram_range = (1,4))
tfs = tfidf.fit_transform(corpus.values())
print(tfidf.get_feature_names())
#print(tfs.toarray())
print(pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names(), index=['doc1','doc2','doc3']))
df=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names(), index=['doc1','doc2','doc3'])

##includes all from document (except for single character)
tfidf = TfidfVectorizer()
tfs = tfidf.fit_transform(corpus.values())
print(tfidf.get_feature_names())
#print(tfs.toarray())
print(pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names(), index=['doc1','doc2','doc3']))
df=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names(), index=['doc1','doc2','doc3'])

#To get the words per document which is not 'ZERO' count
index=['doc1','doc2','doc3']
for i,word in enumerate(index):
    print(df.iloc[i])
    temp_list=df.iloc[i].to_dict()
    var=[ j for j in temp_list if temp_list[j]>0 ] 
    print("In this document the words with value>0 are",var)
 
    
################################TESTING################    
corpus = {1: "game of everlasting LEARNING ", 2: "worth living", 3: "Never stop learning"}     
df=pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names(), index=['doc1','doc2','doc3'])


#The lenght of the text
sum([len(x) for x in combi['tweet']]) 
#Remove all words with size less than 3
combi['tidy_tweet']=combi['tidy_tweet'].apply(lambda x:  ' '.join([w for w in x.split() if len(w) > 3]))




all_words = ' '.join([text for text in combi['tidy_tweet']])


#Non-Racist/Sexist Tweets top 10 words
a = nltk.FreqDist(HT_regular)  #gives the WORD and its COUNT OF REPETITIONS in dictionary format

#Create dataframe for dictionary
d=pd.DataFrame()
d['Hashtag']=a.keys()
d['Count']=a.values()

# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Racist/Sexist Tweets top 10 words
b = nltk.FreqDist(HT_negative)
e=pd.DataFrame()
e['Hashtag']=b.keys()
e['Count']=b.values()
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Just example for One-GRAM and BI-Grams
#ONEgram_vectorizer = CountVectorizer(stop_words='english',token_pattern=r'\b\w+\b', min_df=1)
#analyze = ONEgram_vectorizer.build_analyzer()
#analyze('Bi-grams are cool!') 
#
#bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
#analyze = bigram_vectorizer.build_analyzer()
#analyze('Bi-grams are cool!') 




