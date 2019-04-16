

####https://pandas.pydata.org/pandas-docs/stable/visualization.html (LINK for GOOD GRAPHS)

import matplotlib.pyplot as plot
from scipy.io import wavfile

# Read the wav file (mono)
samplingFrequency, signalData = wavfile.read('/Users/raghuram.b/Desktop/speech/audiofiles/WaveSample1.wav')
# Plot the signal read from wav file
#plot.subplot(500)
plot.title('Spectrogram of audio file')
plot.plot(signalData)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
plot.show()

################################################################
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
spf = wave.open('/Users/raghuram.b/Desktop/speech/plumbing.wav','r')
#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()
#If Stereo
if spf.getnchannels() == 2:
    print('Just mono files')
    sys.exit(0)

Time=np.linspace(0, len(signal)/fs, num=len(signal))
plt.figure(1)
plt.title('Signal Wave...')
plt.plot(Time,signal)
plt.show()
################################################################
Total emotions graph
################################################################
import matplotlib.pyplot as plt
import pandas 
import numpy as np
df=pandas.read_csv("/Users/raghuram.b/Desktop/speech/transcripts/emotion_transcript_WaveSample1.csv")

label=df['emotion'].unique()
emotion=dict(df['emotion'].value_counts())

label_counts=[]
for i in range(len(label)):
   label_counts.append(emotion[label[i]])

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, label_counts,color='blue')
    plt.xlabel('Emotion', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title('Frequency of Emotions in audio file')
    plt.show()

#plot the graph    
plot_bar_x()
################################################################
Every AUDIO FILE 
################################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df=pd.read_csv("/Users/raghuram.b/Desktop/speech/transcripts/emotion_transcript_WaveSample1.csv")

label = ['AudioFile1', 'AudioFile2', 'AudioFile3', 'AudioFile4','AudioFile5','AudioFile6','AudioFile7','AudioFile8']
df['label']=label
duration = [941,854,300,900,100,30,445,780]
df['duration']=duration

#create colours
emotion_dict={'Excited': 'yellow', 'Angry': 'red', 'Bored': 'cyan', 'Happy': 'green', 'Fear': 'magenta', 'Neutral': 'magenta', 'Sad': 'black'}
#assign colour to emotions
def emotion_color(x):
   return emotion_dict[x]
df['emotion_color']=df['emotion'].apply(lambda x: emotion_color(x))
   


#plot graph
index = np.arange(len(label))

for idx in index:
    plt.barh(idx, df.duration[idx],color=df.emotion_color[idx])   
    #plt.bar(index, duration,color=df['emotion_color'])
    plt.xlabel('AudioFiles', fontsize=5)
    plt.ylabel('Duration of Audio Files', fontsize=5)
    plt.yticks(index, label, fontsize=5, rotation=30)
    plt.title('[POC]Emotional values of Audio files')
 

plt.show()


########################
With Legends
########################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df=pd.read_csv("/Users/raghuram.b/Desktop/speech/transcripts/emotion_transcript_WaveSample1.csv")

label = ['AudioFile1', 'AudioFile2', 'AudioFile3', 'AudioFile4','AudioFile5','AudioFile6','AudioFile7','AudioFile8']
df['label']=label
duration = [941,854,300,900,100,30,445,780]
df['duration']=duration

#create colours
emotion_dict={'Excited': 'yellow', 'Angry': 'red', 'Bored': 'cyan', 'Happy': 'green', 'Fear': 'magenta', 'Neutral': 'magenta', 'Sad': 'black'}
#assign colour to emotions
def emotion_color(x):
   return emotion_dict[x]
df['emotion_color']=df['emotion'].apply(lambda x: emotion_color(x))
   


#plot graph
index = np.arange(len(label))

for idx in index:
    plt.barh(idx, df.duration[idx],color=df.emotion_color[idx])   
    #plt.bar(index, duration,color=df['emotion_color'])
    plt.xlabel('AudioFiles', fontsize=5)
    plt.ylabel('Duration of Audio Files', fontsize=5)
    plt.yticks(index, label, fontsize=5, rotation=30)
    plt.title('[POC]Emotional values of Audio files')

emotion=df['emotion']
plt.legend(emotion,loc=1)
plt.show()
  
df[['emotion','duration','emotion_color']]




