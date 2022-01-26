# from view_spectrogram import spectro
import scipy
from scipy.io import wavfile
if __name__== "__main__":
    import librosa, librosa.display
    import numpy as np
    # import matplotlib.pyplot as plt
    # import torch
    file = "2.wav"
    file2 = "1.wav"
    '''able to convert any .wav file to spectrogram in pytorch and back''' 


    # torch.set_printoptions(precision=10)
    #numpy array                

    signal,sr = librosa.load(file,mono=False,sr=44100,duration=None)
    # print(len(signal)/44100)
    # print(signal.shape)
    signal_l = signal[0]
    signal_r = signal[1]

    stft_l = librosa.core.stft(signal_l,hop_length=1024,n_fft=4096,center=True) #overlap
    stft_r = librosa.core.stft(signal_r,hop_length=1024,n_fft=4096,center=True) #overlap
   
   
    stft_stereo = np.dstack((stft_l,stft_r))
    print(stft_stereo.shape)

    source_l = librosa.istft(stft_l)
    source_r = librosa.istft(stft_r)

    source_stereo = np.vstack((source_l,source_r)).transpose() 

    print(source_stereo.shape) 
    
    wavfile.write("source_l.wav",44100,source_l) #
    wavfile.write("source_r.wav",44100,source_r) #
    wavfile.write("source_stereo.wav",44100,source_stereo) #
    '''
    print(signal.shape)
    stft = librosa.core.stft(signal,hop_length=1024,n_fft=4096,center=True) #overlap
    print(stft.shape)
    stft = librosa.core.stft(signal[int(len(signal)/2):],hop_length=1024,n_fft=4096,center=True) #overlap
    print(stft.shape)

    print(5/0)
    magn, phase = librosa.magphase(stft)
    stft_mag = np.abs(stft)
    # print(magn == stft_mag)
    signal2,sr2 = librosa.load(file2,sr=44100)
    # print(len(signal)/44100)
    # print(signal.shape)
    stft2 = librosa.core.stft(signal2,hop_length=1024,n_fft=4096,center=True) #overlap
    magn2, phase2 = librosa.magphase(stft2)

 
    istft = librosa.istft(stft2) # correct
    istft_c = librosa.istft(stft)
    dif1 = librosa.griffinlim(magn)
    dif2 = librosa.istft(magn)
    
    print(np.mean(np.abs(np.subtract(istft_c,dif1))))
    print(np.mean(np.abs(np.subtract(istft_c,dif2))))
    print(np.mean(np.abs(np.subtract(istft_c,signal2[:220160]))))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    librosa.display.waveshow(signal2, sr=sr, color='b', ax=ax[0])
    ax[0].set(title='Original', xlabel=None)
    ax[0].label_outer()
    librosa.display.waveshow(dif1, sr=sr, color='g', ax=ax[1])
    ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
    ax[1].label_outer()
    librosa.display.waveshow(dif2, sr=sr, color='r', ax=ax[2])
    ax[2].set_title('Magnitude-only istft reconstruction')
    plt.show()
    wavfile.write("griffin_wav2.wav",44100,dif1) #
    wavfile.write("griffin.wav",44100,dif2)       #
    # print(np.abs(np.subtract( phase[200], phase2[200])))
    # print((phase2[5]))
    print(5/0)
'''
'''

    # print(magn[0])
    # print(phase[0])

    # print(5/0)
    print("lenght",stft.shape, len(stft)/4)
    stft_a = stft
    stft_m = np.abs(np.mean(stft_a[:1300:],axis=1))


    # print("mean",np.mean(stft_m)*100)
 
    istft = librosa.core.istft(stft,hop_length=1024)
    print(stft.shape)
    print(istft.shape)
    
    wavfile.write("testOut.wav",44100,istft)
    file = "testOut.wav"
    
    print ( "new file")
    signal,sr = librosa.load(file,sr=44100)
    print(len(signal)/44100)
    print(signal.shape)
    stft = librosa.core.stft(signal,hop_length=1024,n_fft=4096,center=True) #overlap
    istft = librosa.core.istft(stft,hop_length=1024)
    print(stft.shape)
    print(istft.shape)

 
    # spectrogram = np.abs(stft) #first is frequency second is time
    # print((np.min(spectrogram)))
    # # log spectrogram because that's how humans perceive sound 
    # log_spectrogram = librosa.amplitude_to_db(spectrogram)


    # print(spectrogram.shape)
    # print(log_spectrogram.shape)
    '''