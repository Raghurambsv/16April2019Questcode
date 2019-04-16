#IMPORTING THE PACKAGES
import re
import glob
import pickle
import pandas as pd
import os
         
part='Final_Model_'


#ZZ-VDSHP        684
#ZZ-VRZZZ        261
#ZZ-VDLAY        209
#ZZ-QRCRI        183
#ZZ-VDZZZ         141 
#EM-QRCRI        100 

#Load the Pickled  Model to respective Target variables
loaded_model_origin = pickle.load(open('./PickleFiles/'+part+'Originator.pkl', 'rb'))
loaded_model_disc = pickle.load(open('./PickleFiles/'+part+'Discipline.pkl', 'rb')) 
loaded_model_doctype = pickle.load(open('./PickleFiles/'+part+'Document Type.pkl', 'rb'))
loaded_model_docsubtype = pickle.load(open('./PickleFiles/'+part+'Document Subtype.pkl', 'rb'))
loaded_model_filepath = pickle.load(open('./PickleFiles/'+part+'FilePath.pkl', 'rb'))



#Load the Pickled LabelEncoder files & Decode them respectively
Label_Org=pickle.load(open('./PickleFiles/labelencoder/'+part+'Orginator_LableEncoder.pkl','rb'))
Label_Disc=pickle.load(open('./PickleFiles/labelencoder/'+part+'Discipline_LableEncoder.pkl','rb'))
Label_Dtype=pickle.load(open('./PickleFiles/labelencoder/'+part+'DocType_LableEncoder.pkl','rb'))
Label_DStype=pickle.load(open('./PickleFiles/labelencoder/'+part+'DocSubtype_LableEncoder.pkl','rb'))
Label_FilePath=pickle.load(open('./PickleFiles/labelencoder/'+part+'FilePath_LableEncoder.pkl','rb'))

 

#Cleaning/Preprocessing the Data
def clean_text(text):
    text=str(text.encode("utf-8"))
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return str(text) 


#Predicting the TextFile Category  for the Fresh Textfiles   
df = pd.DataFrame(columns=['FileName','Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath','Originator Predicted','Discipline Predicted','Document Type Predicted','Document Subtype Predicted','FilePath Predicted'])    
df1 = pd.DataFrame(columns=['FileName','Originator', 'Discipline', 'Document Type', 'Document Subtype','FilePath','Originator Predicted','Discipline Predicted','Document Type Predicted','Document Subtype Predicted','FilePath Predicted'])

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
            df1['Document Type']=loaded_model_doctype.predict(data)
            df1['Document Subtype']=loaded_model_docsubtype.predict(data)
            df1['FilePath']=loaded_model_filepath.predict(data)
            
            df1['Originator Predicted']=df1['Originator'].apply(lambda x: 'Others' if x > 1  else Label_Org.inverse_transform(df1['Originator']))
            df1['Discipline Predicted']=df1['Discipline'].apply(lambda x: 'Others' if x > 1  else Label_Disc.inverse_transform(df1['Discipline']))
            df1['Document Type Predicted']=df1['Document Type'].apply(lambda x: 'Others' if x > 2  else Label_Dtype.inverse_transform(df1['Document Type']))
            df1['Document Subtype Predicted']=df1['Document Subtype'].apply(lambda x: 'Others' if x > 3  else Label_DStype.inverse_transform(df1['Document Subtype']))
            df1['FilePath Predicted']=Label_FilePath.inverse_transform(df1['FilePath'].values.astype(int))
            df1['FileName']=txtfile.split('/')[-1]
            df=df.append(df1,sort=None)
            count=count+1
print("##################################")
print("Total No of files processed :",count)
print("##################################")
      
#df['FilePath']=loaded_model_filepath.predict(data)
#df['FilePath Predicted']=Label_FilePath.inverse_transform(df['FilePath'])      
#Reset the index             
df.reset_index(drop=True,inplace=True)   



dfinal = pd.DataFrame(columns=['FileName','Originator Predicted','Discipline Predicted','Document Type Predicted','Document Subtype Predicted','FilePath Predicted'])


def output_extract(text):
    text=str(text)
    text=text.replace("[","")
    text=text.replace("]","")
    text=text.replace("'","")
    return str(text)

dfinal['FileName']=df['FileName']
dfinal['Originator Predicted']=df['Originator Predicted'].apply(output_extract)
dfinal['Discipline Predicted']=df['Discipline Predicted'].apply(output_extract)
dfinal['Document Type Predicted']=df['Document Type Predicted'].apply(output_extract)
dfinal['Document Subtype Predicted']=df['Document Subtype Predicted'].apply(output_extract)
dfinal['FilePath Predicted']=df['FilePath Predicted'].apply(output_extract)
        

dfinal.dropna(how='any')


dfinal.to_csv('./Fresh_Text_Files/'+part+'Prediction_output.csv',index=False,encoding="utf-8")









