import speech_recognition as sr
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
r=sr.Recognizer()


#Speech to Emotion for all files
import glob2
filenames=glob2.glob("/Users/raghuram.b/Desktop/speech/testfiles/*")
label_list=[]
duration_list=[]
emotion_color_list=[]
emotion_list=[]

for file in filenames:
    print('Processing for file:',file)
    #Getting filename
    sub=file.split('/')
    sub=sub[-1]
    label_list.append(sub)
    sub=sub.split('.')
    sub=sub[-2]
  
    #Calculate the length of audio and process every 10 secs of it
    import wave
    def get_duration_wav(wav_filename):
        f = wave.open(wav_filename, 'r')
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        f.close()
        return duration
    
    duration=round(get_duration_wav(file))
    duration_list.append(duration)
    print(duration)
    length=round(duration/30)
    print(length)
    
    
    #convert the speech to text
    text_list=[]
    with open("/Users/raghuram.b/Desktop/speech/EmotionPOC-4312f9e6eb3d.json") as f:
        GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()
        
    harvard = sr.AudioFile(file)
    #fixing lenght to just 3 parsers irrespective of file length
    length=4
    for i in range(length):
            j=i * 25
            with harvard as source:
                    audio = r.record(source, duration=25,offset=j)
                    text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)    
                    print(text)
                    text_list.append(text)
                    
    df=pd.DataFrame(text_list, columns=['body_text'])
    
    df.to_csv('/Users/raghuram.b/Desktop/speech/transcripts/transcript_'+sub+'.csv')
    
    
    #process emotions of the speech
    import paralleldots
    
    api_key   = "s5ftlBJOKZTg4CbAan5E4JuE8pNZ46XsRuDUY1VNDZM"
    #text      = "good afternoon thank you for calling Professional Plumbing yes hi I need a plumber to come out to my home as soon as possible but that green is a g r e e n o g r e e n e thank you mr. green that's Greene mr. green there have your area code and telephone number please 265-555-1338 thank you sir that's 626-555-1338 must agree three-bedroom two-and-a-half-bath two of my toilets are backing up and I don't know so so I need a plumber I need a plumber to come out as soon as possible definitely serve will be"
    df=pd.read_csv('/Users/raghuram.b/Desktop/speech/transcripts/transcript_'+sub+'.csv')
    #df=df.iloc[2:4]
    
    
    lenght=len(df)
    text=[]
    for i in range(lenght):
        text.append(df.body_text[i])
    
    #Generate emotion
    temp=dict(paralleldots.emotion( text ))
    emotion=temp['emotion']['emotion']   
    print(emotion) 
    emotion_list.append(emotion)
    
    
    #Generate DF of one Emotion for text
    df1=pd.DataFrame(columns=['emotion','body_text','duration'])
    df1.loc[0]= [emotion,text,duration]
    
     #Combining emotions to similar colour code[FOR Dataframe]
    df1.replace(to_replace=r'Sarcasm', value='Neutral', regex=True,inplace=True)
    df1.replace(to_replace=r'Bored', value='Neutral', regex=True,inplace=True)
    df1.replace(to_replace=r'Excited', value='Happy', regex=True,inplace=True)
    
    #Combining emotions to similar colour code [FOR LIST]
    emotion_list = [word.replace('Sarcasm','Neutral') for word in emotion_list]
    emotion_list = [word.replace('Bored','Neutral') for word in emotion_list]
    emotion_list = [word.replace('Excited','Happy') for word in emotion_list]
    
    df1.to_csv('/Users/raghuram.b/Desktop/speech/transcripts/emotion_transcript_'+sub+'.csv')
    
    
#create colours
emotion_dict={ 'Angry': 'red', 'Happy': 'olivedrab', 'Fear': 'orange', 'Neutral': 'wheat', 'Sad': 'black'}
#assign colour to emotions
for i in emotion_list:
    emotion_color_list.append(emotion_dict[i])

#label_list=['file1','file2']
index = np.arange(len(label_list))


#plt.barh(index, df.duration[idx],color=df.emotion_color[idx])   
plt.bar(index, duration_list,color=emotion_color_list,edgecolor='black')
#plt.xlabel('AudioFiles', fontsize=10)
plt.ylabel('Duration --->', fontsize=12)
plt.xticks(index, label_list, fontsize=10, rotation=20)
plt.title('Emotions / Audio files')
#plt.legend(emotion_dict)
plt.show()
    

paralleldots.usage()



from matplotlib.pyplot import *


subplot(221)
plot([1,20,3], label="Angry" ,color='red')
plot([3,2,1], label="Neutral",color='moccasin')
plot([3,2,1], label="Happy",color='olivedrab')
plot([1,2,3], label="Fear",color='orange')

plot([1,2,3], label="Sad",color='black')

legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
frame = legend.get_frame()
frame.set_facecolor('green')

show()
  
