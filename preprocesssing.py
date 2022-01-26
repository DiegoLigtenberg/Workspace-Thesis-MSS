''' 
1= load a file
2- pad the signal (if necessary)
3- extracting log spectrogram from signal
4- normalise spectrogram
5- save the normalised spectrogram

Preprocessing Pipeline
'''

# https://www.youtube.com/watch?v=O04v3cgHNeM&ab_channel=ValerioVelardo-TheSoundofAI&loop=0

import librosa
import numpy as np
import os
# from view_spectrogram import spectro
import musdb
from sklearn.cluster import spectral_clustering

# this function splits the music tracks on alphabetical order instead of order in directory
mus_train = musdb.DB("database_wav",subsets="train", split='train',download=False,is_wav=True)

DATASET_DIR = "database_chunk"
SPECTROGRAM_SAVE_DIR = "train_spectrogram"


class Loader():
    '''Load is responsible for loading an audio file.'''
    def __init__(self,sample_rate,mono,duration=None):
        self.sample_rate = sample_rate
        self.mono = mono
        self.duration = duration
    
    def load(self,file_path):
        signal = librosa.load(file_path,sample_rate=self.sample_rate,mono=self.mono,duration=self.duration)[0]
        return signal


train_data= []
for root,_,files in os.walk(DATASET_DIR):
    for file in files:
        file_path = os.path.join(root,file)
        if ("mixture" in file_path):
            train_data.append(load(file_path))

print(len(train_data))
          
   

class Padder:
    '''responsible to apply zero padding to an array'''

    def __init__(self,mode="constant"):
        self.mode = mode
    
    def left_pad(self,array,num_missing_items):
        padded_array = np.pad(array,num_missing_items,0,mode=self.mode)
        return padded_array

    def right_pad(self,array,num_missing_items): 
        padded_array = np.pad(array,0,num_missing_items,mode=self.mode)
        return padded_array

class LogSpectroGramExtractor:
    '''extracts logspectrogram (in dB) from a time-series (audio) signal'''
    
    def __init__(self,frame_size,hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self,signal):

        stft = librosa.stft(signal,n_fft=self.frame_size,hop_length=self.hop_length)[:-1] #dimensions = (1+ (frame_size/2)  , num_frames)  1024 -> 513 -> 512 ([:-1])
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
 

class MinMaxNormaliser:
    pass

class Saver:
    pass


class PreprocessingPipeline:
    '''
    PrprocesspingPipeline processes audio file in a directory. applying the following steps
    1- load a file
    2- pad the signal (if necessary)
    3- extracting log spectrogram from signal
    4- normalise spectrogram
    5- save the normalised spectrogram
    '''

    def __init__(self):
        self.loader = None
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}

    def process(self,audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)

    def _process_file(self,file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_neccessary(signal):
            signal = self.__apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature)

class Augmentation():
    '''this class is responsible for performing certain data augmentation techniques'''
    def __init__(self) -> None:
        pass

