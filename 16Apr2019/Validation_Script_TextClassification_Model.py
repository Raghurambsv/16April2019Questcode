import re
import glob
import sys
import os.path
import re
import glob
import pickle
import pandas as pd
import os
         
part=''

fn = sys.argv[1]
if os.path.exists(fn):
    path=os.path.realpath(fn)

#Originators
#-----------
#ZZ	3510
#EM	427


#Discipline	
#-----------
#V	    2192
#Q	    512
#D    	268
#P	    187
#M	    176
#L	    145
#B	    128
#U 	    114
#Others	215


#DocType & DocSubType	
#--------------------
#DSHP	760
#RZZZ	327
#RCRI	292
#DLAY	232
#DISO	123
#DPID	114
#XXER	111
#RTQM	93
#Others	1885


#Load the Pickled  Model to respective Target variables
loaded_model_origin = pickle.load(open('./PickleFiles/'+part+'Originator.pkl', 'rb'))
loaded_model_disc = pickle.load(open('./PickleFiles/'+part+'Discipline.pkl', 'rb')) 
loaded_model_doc_type_and_subtype = pickle.load(open('./PickleFiles/'+part+'combi.pkl', 'rb'))




#Load the Pickled LabelEncoder files & Decode them respectively
Label_Org=pickle.load(open('./PickleFiles/labelencoder/'+part+'Orginator_LableEncoder.pkl','rb'))
Label_Disc=pickle.load(open('./PickleFiles/labelencoder/'+part+'Discipline_LableEncoder.pkl','rb'))
Label_DTST=pickle.load(open('./PickleFiles/labelencoder/'+part+'Type_and_Subtype_LableEncoder.pkl','rb'))


 

#Cleaning/Preprocessing the Data
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

#Predicting the TextFile Category  for the Fresh Textfiles
df = pd.DataFrame(columns=['Originator', 'Discipline', 'Doc_Type_and_SubType', 'Originator Predicted','Discipline Predicted','TypeSubtype Predicted'])    
df1 = pd.DataFrame(columns=['Originator', 'Discipline', 'Doc_Type_and_SubType', 'Originator Predicted','Discipline Predicted','TypeSubtype Predicted'])
    

print("\nThe text files considered for this run are as below:")
print("------------------------------------------------------")
count=0
for txtfile in os.listdir("./Fresh_Text_Files/"):
    if txtfile.endswith(".txt"):
       txtfile=os.path.join("./Fresh_Text_Files/", txtfile)
       print(txtfile.split('/')[-1])
       with open(txtfile,'r',encoding="utf-8") as file:
            data=file.read().replace('\n', '')
            data=clean_text(data)
            data=pd.Series(data) 
            df1['Originator']=loaded_model_origin.predict(data)
            df1['Discipline']=loaded_model_disc.predict(data)
            df1['Doc_Type_and_SubType']=loaded_model_doc_type_and_subtype.predict(data)
          
            
            df1['Originator_prob']= max(loaded_model_origin.predict_proba(data).round(2).tolist()[0])
            df1['Discipline_prob']= max(loaded_model_disc.predict_proba(data).round(2).tolist()[0])
            df1['Doc_Type_and_SubType_prob']= max(loaded_model_doc_type_and_subtype.predict_proba(data).round(2).tolist()[0])
                    
      
            df1['Originator Predicted']=df1['Originator'].apply(lambda x: Label_Org.inverse_transform(df1['Originator']))
            df1['Discipline Predicted']=df1['Discipline'].apply(lambda x: 'Others' if x > 7  else Label_Disc.inverse_transform(df1['Discipline']))
            df1['TypeSubtype Predicted']=df1['Doc_Type_and_SubType'].apply(lambda x: 'Others' if x > 7  else Label_DTST.inverse_transform(df1['Doc_Type_and_SubType']))
            df1['FileName']=txtfile.split('/')[-1]
                      
            
            df=df.append(df1,sort=None)
            count=count+1
print("##################################")
print("Total No of files processed :",count)
print("##################################")
      
     
#Reset the index             
df.reset_index(drop=True,inplace=True)   





dfinal = pd.DataFrame(columns=['Original_FileName','Originator Predicted','Discipline Predicted','TypeSubtype Predicted','Filename predicted',' <Originator-Confidence> ',' <Discipline-Confidence> ',' <TypeSubtype-Confidence> '])


def output_extract(text):
    text=str(text)
    text=text.replace("[","")
    text=text.replace("]","")
    text=text.replace("'","")
    return str(text)

dfinal['Original_FileName']=df['FileName']
dfinal['Originator Predicted']=df['Originator Predicted'].apply(output_extract)
dfinal['Discipline Predicted']=df['Discipline Predicted'].apply(output_extract)
dfinal['TypeSubtype Predicted']=df['TypeSubtype Predicted'].apply(output_extract)

dfinal[' <Originator-Confidence> ']=df['Originator_prob']
dfinal[' <Discipline-Confidence> ']=df['Discipline_prob']
dfinal[' <TypeSubtype-Confidence> ']=df['Doc_Type_and_SubType_prob']


dfinal['Filename predicted']=""
for i, row in df.iterrows():
    index=str(i)
    dfinal.at[i,'Filename predicted'] = 'BIRF-'+ dfinal.at[i,'Originator Predicted'] +'-'+ dfinal.at[i,'Discipline Predicted'] +'-'+ dfinal.at[i,'TypeSubtype Predicted']+'-'+index+'.pdf'
  
dfinal.dropna(how='any')

dfinal[' <Originator-Confidence> ']=dfinal[' <Originator-Confidence> '].map(str)
dfinal[' <Originator-Confidence> '] = dfinal[['Originator Predicted', ' <Originator-Confidence> ']].apply(lambda x: '- '.join(x), axis=1)

dfinal[' <Discipline-Confidence> ']=dfinal[' <Discipline-Confidence> '].map(str)
dfinal[' <Discipline-Confidence> '] = dfinal[['Discipline Predicted', ' <Discipline-Confidence> ']].apply(lambda x: '- '.join(x), axis=1)

dfinal[' <TypeSubtype-Confidence> ']=dfinal[' <TypeSubtype-Confidence> '].map(str)
dfinal[' <TypeSubtype-Confidence> '] = dfinal[['TypeSubtype Predicted', ' <TypeSubtype-Confidence> ']].apply(lambda x: '- '.join(x), axis=1)


dfinal.to_csv('./Fresh_Text_Files/'+part+'Prediction_output.csv',index=False,encoding="utf-8")



