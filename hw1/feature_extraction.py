# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 20

def extract_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print i

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        #print file_path
        y, sr = librosa.load(file_path, sr=22050)


        ##### Method 1
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
        
        ##### Method 2 
        """
        # STFT
        S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)

        # power spectrum
        D = np.abs(S)**2

        # mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, 1024, n_mels=40)
        mel_S = np.dot(mel_basis, D)

        #log compression
        log_mel_S = librosa.power_to_db(mel_S)

        # mfcc (DCT)
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=13)
        mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)
        """

        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc)

    f.close();

if __name__ == '__main__':
    extract_mfcc(dataset='train')                 
    extract_mfcc(dataset='valid')                                  
    extract_mfcc(dataset='test')



