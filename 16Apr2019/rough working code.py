




import re
import os
import pandas as pd
import csv  



##Open the file to be parsed
with open("/Users/raghuram.b/Desktop/Exxon/Getquotes.txt",'r') as file:
    content=file.readlines()
    print(content)

#####################Database & Tablename#####################
list1=[]
for line in content:
    result=re.match(r"^.*\"(\w+\s*\w*)\".*\"(\w+)\".*",line)

    if result == None:
        pass
    else:
        with open("/Users/raghuram.b/Desktop/Exxon/result.txt",'a') as file:
            file.writelines(result.group(1)+"\n")
        print(result.group(1))
        list1.append(result.group(1))



#getting unique list of xls search items
s=[]        
for i in list1:
       if i not in s:
          s.append(i)
s=[str.lower() for str in s]
#s=s[:2]          
#search inside indivual files          
import glob2
import json
#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/BIRF-EM-DDZZZ-510*")
filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/*")
name=[]
count=0

for filename in filelist:
    print(filename)
#Creating dictionary with value zero for all pattern's    
    dicta = {}
    for i in range(0, len(s)):
        dicta[s[i]] = 0
    with open(filename,'r') as file:
        content=file.readlines()
        
    #convert content to lower case    
    content=" ".join(content)
    content=content.lower()        
        

    for pattern in s:
                 
        pattern=pattern.lower()
        with open("/Users/raghuram.b/Desktop/Exxon/patternfiles/"+pattern+".txt",'a') as file:            
            if content.find(pattern) != -1: 
                file.writelines(filename.split("/")[-1]+"\n") 
                
                for i in dicta.keys():
                    if (i == pattern):
                        count=dicta[pattern]+1
                        dicta[pattern]=count
#                        print(pattern,count,json.dumps(dicta))
#    finaldict=sorted(dicta.items(), key=lambda x: x[1],reverse=True) 
    finaldict = {k:v for k, v in dicta.items() if v != 0}   
    with open("/Users/raghuram.b/Desktop/dictionary.txt",'a') as file:
        file.writelines("Filename====>"+filename.split("/")[-1]+"         "+"patterns ====>"+json.dumps(finaldict)+"\n")
            
 
       

import glob2
#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/BIRF-EM-DDZZZ-510*")
filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/*")
totalfiles=[]
for i in filelist:       
    totalfiles.append(i.split("/")[-1]) 
print("Total files text files from quest :",len(totalfiles))      

patfilelist=[]    
patlist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/patternfiles/*")
for i in patlist:
        with open(i,'r') as file:
            content=file.readlines()
            for j in content:
                patfilelist.append(j)
#Remove extra "\n"
patfilelist = [w.replace('\n', '') for w in patfilelist]       
#Remove duplicates
patfilelist=set(patfilelist)  
print("Files captured from pattern in xls :",len(patfilelist))          
    

missing=set(totalfiles)-set(patfilelist)
    
print("Total files missed out on classification",len(missing))  

#converting back to list
missing=list(missing)
missing=missing[:5]
      
    
###ONLY 957 files  (which is relevant as per XLS) #############
import glob2
#filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/TXTs/BIRF-EM-DDZZZ-510*")
filelist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/pdfs/*")
totalfiles=[]
for i in filelist:       
    totalfiles.append(i.split("/")[-1])
#Remove pdf/PDF extension
totalfiles=[ w.replace('.pdf','') for w in totalfiles]
totalfiles=[ w.replace('.PDF','') for w in totalfiles]      
print("Total text files from quest[SMALL FOLDER] :",len(totalfiles))      

patfilelist=[]    
patlist=glob2.glob("/Users/raghuram.b/Desktop/Exxon/patternfiles/*")
for i in patlist:
        with open(i,'r') as file:
            content=file.readlines()
            for j in content:
                patfilelist.append(j)
#Remove extra "\n"
patfilelist = [w.replace('\n', '') for w in patfilelist] 
#Remove txt extension 
patfilelist = [w.replace('.txt', '') for w in patfilelist]      
#Remove duplicates
patfilelist=set(patfilelist)  
print("Files captured from pattern in xls[FOR ALL 3589 files] :",len(patfilelist))          
    

missing=set(totalfiles)-set(patfilelist)
    
print("Total files missed out on classification",len(missing))  
print("watever is missed is mostly DIAGRAMS/unidentified TEXT")



                     
