#max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#max_df = 25 means "ignore terms that appear in more than 25 documents".
#min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#min_df = 5 means "ignore terms that appear in less than 5 documents".

from sklearn.feature_extraction.text import CountVectorizer

#vectorizer = CountVectorizer()
#bag_of_words=vectorizer.fit_transform(corpus)
#X = vectorizer.fit_transform(allsentences)
#print(X.toarray())
corpus=[]
for j in range(0,9):
    for i in df.content[j].split(' '):
        corpus.append(i)
        
corpus=[ch for ch in corpus if len(ch) > 4]     
vec = CountVectorizer().fit(corpus)
bag_of_words = vec.transform(corpus)
sum_words = bag_of_words.sum(axis=0) 

words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()] #Feature names with there no of counts
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True) #Feature names with there no of counts (In reverse order)
print(words_freq[:30]) #Top 30 words


vocab = vec.get_feature_names() #Only Feature Names
print(vocab)
print(vec.vocabulary_) #Feature names with there index
print(vec.transform(corpus).toarray()) #it shows matrix of per documents (if word present its '1' else its '0')


###################################### COUNTVECTORIZER WORD FREQUENCY PROGRAM ENDED ###########################

#Sorting based on some function (like not alphabetical order but on length of string)
L = ["cccc", "b", "dd", "aaa"] 
print ("Sort with len :", sorted(L, key = len))
print(sorted(L))

#Create your own vocab in Countvectorizer by omiting unwanted words
list_of_words_to_omit = ["xxxdd", "cat","nine",'everywhere','seven','three','the']
#list_of_words_to_omit = frozenset(["xxxdd", "cat","nine",'everywhere','seven','three','the'])
corpus = ['one two three everywhere', 'four five six everywhere', 'seven eight nine everywhere','the of','for','xxxdd','cat nine']
cv = CountVectorizer(min_df=1, max_df=1.0, lowercase=True) 
#cv = CountVectorizer(min_df=1, max_df=1.0,ngram_range=(1,2), lowercase=True) 
#cv = CountVectorizer(min_df=1, max_df=1.0,ngram_range=(1,2), lowercase=True,stop_words='english') 
#cv = CountVectorizer(min_df=1, max_df=1.0,ngram_range=(1,2), lowercase=True,stop_words=list_of_words_to_omit) 
X = cv.fit_transform(corpus)
vocab = cv.get_feature_names()
print(vocab)
print(X.toarray())


words_freq =sorted(words_freq, key = lambda x: x[1], reverse=False)

#https://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/
# It has countvecotriser....tfidf....pipeline....classfication metric report....pipeline prediction

#for num,message in enumerate(messages[:10]):
#    print(num,message)
#    print ('\n')
#    
#messages.groupby('labels').describe()
#
#messages['length'] = messages['message'].apply(len)
#messages.head()
#
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#messages['length'].plot(bins=50,kind = 'hist')
#messages[messages['length'] == 910]['message'].iloc[0]
#messages.hist(column='length',by ='labels',bins=50,figsize = (10,4))    

