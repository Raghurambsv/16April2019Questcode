import speech_recognition as sr
import pandas as pd
import re
r=sr.Recognizer()

#Calculate the length of audio and process every 10 secs of it
import wave
def get_duration_wav(wav_filename):
    f = wave.open(wav_filename, 'r')
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()
    return duration

duration=get_duration_wav('/Users/raghuram.b/Desktop/speech/audiofiles/WaveSample1.wav')
print(duration)
length=round(duration/20)
print(length)


#convert the speech to text
text_list=[]
with open("/Users/raghuram.b/Desktop/speech/EmotionPOC-4312f9e6eb3d.json") as f:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()
    
harvard = sr.AudioFile('/Users/raghuram.b/Desktop/speech/audiofiles/WaveSample1.wav')
for i in range(length-1):
        i=i * 20
        with harvard as source:
                audio = r.record(source, duration=10,offset=i)
                text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)    
                print(text)
                text_list.append(text)
                
df=pd.DataFrame(text_list, columns=['body_text'])
df.to_csv('//users//raghuram.b//Desktop//transcript_WaveSample1.csv')


#process emotions of the speech
import paralleldots


api_key   = "xZknTEd2sAcJ0I6o3d7RrMLdIRtEar0aRlgwMdqyvpc"
#text      = "good afternoon thank you for calling Professional Plumbing yes hi I need a plumber to come out to my home as soon as possible but that green is a g r e e n o g r e e n e thank you mr. green that's Greene mr. green there have your area code and telephone number please 265-555-1338 thank you sir that's 626-555-1338 must agree three-bedroom two-and-a-half-bath two of my toilets are backing up and I don't know so so I need a plumber I need a plumber to come out as soon as possible definitely serve will be"
df=pd.read_csv("//users//raghuram.b//Desktop//transcript_WaveSample1.csv")
#df=df.iloc[2:4]
paralleldots.set_api_key( api_key )
print( "API Key: %s" % paralleldots.get_api_key() )

def emotion_extract(x):
    temp=dict(paralleldots.emotion( x ))
    return temp['emotion']['emotion']

def emotion_body_extract(x):
    temp=dict(paralleldots.emotion( x ))
    return temp  

df['emotion']=df.body_text.apply(lambda x: emotion_extract(x))
df['emotion_body']=df.body_text.apply(lambda x: emotion_body_extract(x))
del df['Unnamed: 0']
df.to_csv('//users//raghuram.b//Desktop//paralleldot.csv')

paralleldots.usage()



#Google
#=======
#AIzaSyAWKGTFG8zY-G3PrdA7bXJkWrXT2FcrRTs
#
#paralledots
#============
#xZknTEd2sAcJ0I6o3d7RrMLdIRtEar0aRlgwMdqyvpc
#
#pip install urllib3[secure]
#
#
#
#json key
#=========
#Emotion POC-45e15e6d15aa.json
#
#sudo apt-get install python3-pyaudio
#Or
#brew update
#brew install portaudio
#brew link --overwrite portaudio
#pip install pyaudio
#
#
#pip install google-api-python-client
#pip install --force-reinstall google-api-python-client
#conda install -c conda-forge google-api-python-client
#
#pip install --upgrade google-cloud-speech
#
#google-api-python-client==1.6.4
#httplib2==0.10.3
#oauth2client==4.1.2
#pyasn1==0.4.2
#pyasn1-modules==0.2.1
#rsa==3.4.2
#six==1.11.0
#SpeechRecognition==3.8.1
#tqdm==4.19.5
#uritemplate==3.0.0



