#IMPORTING THE PACKAGES
import re
import glob
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

#Load the Pickled  Model to respective Target variables
loaded_model_origin = pickle.load(open('./PickleFiles/Originator.pkl', 'rb'))
loaded_model_disc = pickle.load(open('./PickleFiles/Discipline.pkl', 'rb')) 
loaded_model_doctype = pickle.load(open('./PickleFiles/Document Type.pkl', 'rb'))
loaded_model_docsubtype = pickle.load(open('./PickleFiles/Document Subtype.pkl', 'rb'))
loaded_model_filepath = pickle.load(open('./PickleFiles/FilePath.pkl', 'rb'))


#Load the Pickled LabelEncoder files & Decode them respectively
Label_Org=pickle.load(open('./PickleFiles/Orginator_LableEncoder.pkl','rb'))
Label_Disc=pickle.load(open('./PickleFiles/Discipline_LableEncoder.pkl','rb'))
Label_Dtype=pickle.load(open('./PickleFiles/DocType_LableEncoder.pkl','rb'))
Label_DStype=pickle.load(open('./PickleFiles/DocSubtype_LableEncoder.pkl','rb'))
Label_Filepath=pickle.load(open('./PickleFiles/FilePath_LableEncoder.pkl','rb'))
 

#Cleaning/Preprocessing the Data
def clean_text(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text 


#Predicting the TextFile Category  for the Fresh Textfiles   
df = pd.DataFrame(columns=['FileName','Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath'])    
df1 = pd.DataFrame(columns=['FileName','Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath'])
for txtfile in glob.glob('./Fresh_Text_Files/*'):   
    with open(txtfile,'r') as file:
        data=file.read().replace('\n', '')
        data=clean_text(data)
        data=pd.Series(data)         
        df1['FileName']=txtfile.split('/')[-1]
        df1['Originator']=loaded_model_origin.predict(data)
        df1['Discipline']=loaded_model_disc.predict(data)
        df1['Document Type']=loaded_model_doctype.predict(data)
        df1['Document Subtype']=loaded_model_docsubtype.predict(data)
        df1['FilePath']=loaded_model_filepath.predict(data)
        df=df.append(df1,sort=None)
   
#Drop any Null columns and reset index
df=df.dropna(how='any')
#df.drop('index',axis=1,inplace=True)
df.reset_index(inplace=True)

def decode(text):
    text=text.replace("b","")
    text=text.replace("'","")
    return text

df['Originator Predicted']=Label_Org.inverse_transform(df['Originator'].values.astype(int))
df['Originator Predicted']=df['Originator Predicted'].apply(decode)
df['Discipline Predicted']=Label_Disc.inverse_transform(df['Discipline'].values.astype(int))
df['Discipline Predicted']=df['Discipline Predicted'].apply(decode)
df['Document Type Predicted']=Label_Dtype.inverse_transform(df['Document Type'].values.astype(int))
df['Document Type Predicted']=df['Document Type Predicted'].apply(decode)
df['Document Subtype Predicted']=Label_DStype.inverse_transform(df['Document Subtype'].values.astype(int))
df['Document Subtype Predicted']=df['Document Subtype Predicted'].apply(decode)

df['FilePath Predicted']=Label_Filepath.inverse_transform(df['FilePath'].values.astype(int))
df['FilePath Predicted']=df['FilePath Predicted'].apply(decode)

dfinal=df[['FileName','Originator Predicted','Discipline Predicted','Document Type Predicted','Document Subtype Predicted','FilePath Predicted']]

dfinal.dropna(how='any')

dfinal.loc[:,"Originator Predicted"]= dfinal.loc[:,"Originator Predicted"].map(lambda x: x.split(' ')[0])
dfinal.loc[:,"Discipline Predicted"]= dfinal.loc[:,"Discipline Predicted"].map(lambda x: x.split(' ')[0])
dfinal.loc[:,"Document Type Predicted"]= dfinal.loc[:,"Document Type Predicted"].map(lambda x: x.split(' ')[0])
dfinal.loc[:,"Document Subtype Predicted"]= dfinal.loc[:,"Document Subtype Predicted"].map(lambda x: x.split(' ')[0])
#dfinal.loc[:,"FilePath Predicted"]= dfinal.loc[:,"FilePath Predicted"].map(lambda x: x.split(' ')[0])

dfinal.to_csv('./Predicted_output(FreshFiles).csv')









