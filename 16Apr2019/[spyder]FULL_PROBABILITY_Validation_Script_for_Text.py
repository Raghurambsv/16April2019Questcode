#IMPORTING THE PACKAGES
import re
import glob
import pickle
import pandas as pd
import os
         
#part='Final_Model_'
part=''


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

def unzip(x):
    list1=[i for i in x]
    return list1

#Predicting the TextFile Category  for the Fresh Textfiles
df = pd.DataFrame(columns=['Originator', 'Discipline', 'Doc_Type_and_SubType', 'Originator Predicted','Discipline Predicted','Doc_Type_and_SubType Predicted'])    
df1 = pd.DataFrame(columns=['Originator', 'Discipline', 'Doc_Type_and_SubType', 'Originator Predicted','Discipline Predicted','Doc_Type_and_SubType Predicted'])
    

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
            
          
            df1['Originator_prob']= zip(Label_Org.inverse_transform(loaded_model_origin.classes_), loaded_model_origin.predict_proba(data).round(2).tolist()[0])
            df1['Originator_prob']=df1['Originator_prob'].apply(unzip)
            df1['Discipline_prob']=zip(Label_Disc.inverse_transform(loaded_model_disc.classes_), loaded_model_disc.predict_proba(data).round(2).tolist()[0])
            df1['Discipline_prob']=df1['Discipline_prob'].apply(unzip)
            df1['Doc_Type_and_SubType_prob']=zip(Label_DTST.inverse_transform(loaded_model_doc_type_and_subtype.classes_), loaded_model_doc_type_and_subtype.predict_proba(data).round(2).tolist()[0])
            df1['Doc_Type_and_SubType_prob']=df1['Doc_Type_and_SubType_prob'].apply(unzip)
          
            
            df1['Originator Predicted']=df1['Originator'].apply(lambda x: Label_Org.inverse_transform(df1['Originator']))
            df1['Discipline Predicted']=df1['Discipline'].apply(lambda x: 'Others' if x > 7  else Label_Disc.inverse_transform(df1['Discipline']))
            df1['Doc_Type_and_SubType Predicted']=df1['Doc_Type_and_SubType'].apply(lambda x: 'Others' if x > 7  else Label_DTST.inverse_transform(df1['Doc_Type_and_SubType']))
            df1['FileName']=txtfile.split('/')[-1]
            df=df.append(df1,sort=None)
            count=count+1
print("##################################")
print("Total No of files processed :",count)
print("##################################")
      
     
#Reset the index             
df.reset_index(drop=True,inplace=True)   





#dfinal = pd.DataFrame(columns=['Originator Predicted','Discipline Predicted','Doc_Type_and_SubType Predicted'])
dfinal = pd.DataFrame(columns=['FileName(Originator not given by quest)','Originator Predicted','Confidence_Score (Originator)','Discipline Predicted','Confidence_Score (Discipline)','Doc_Type_and_SubType Predicted','Confidence_Score (Doc_Type_and_SubType)'])


def output_extract(text):
    text=str(text)
    text=text.replace("[","")
    text=text.replace("]","")
    text=text.replace("'","")
    return str(text)

dfinal['FileName(Originator not given by quest)']=df['FileName']
dfinal['Originator Predicted']=df['Originator Predicted'].apply(output_extract)
dfinal['Discipline Predicted']=df['Discipline Predicted'].apply(output_extract)
dfinal['Doc_Type_and_SubType Predicted']=df['Doc_Type_and_SubType Predicted'].apply(output_extract)

dfinal['Confidence_Score (Originator)']=df['Originator_prob']
dfinal['Confidence_Score (Discipline)']=df['Discipline_prob']
dfinal['Confidence_Score (Doc_Type_and_SubType)']=df['Doc_Type_and_SubType_prob']


dfinal['Filename predicted']=""
for i, row in df.iterrows():
    index=str(i)
    dfinal.at[i,'Filename predicted'] = 'BIRF-'+ dfinal.at[i,'Originator Predicted'] +'-'+ dfinal.at[i,'Discipline Predicted'] +'-'+ dfinal.at[i,'Doc_Type_and_SubType Predicted']+'-'+index+'.pdf'

dfinal.dropna(how='any')


dfinal.to_csv('./Fresh_Text_Files/'+part+'Prediction_output.csv',index=False,encoding="utf-8")











