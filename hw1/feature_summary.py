# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = './dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 20

def mean_mfcc(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 1000))
    else:
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 200))

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # mean pooling
        temp = np.mean(mfcc, axis=1)
        mfcc_mat[:,i]= np.mean(mfcc, axis=1)
        i = i + 1

    f.close();

    return mfcc_mat
        
if __name__ == '__main__':
    train_data = mean_mfcc('train')
    valid_data = mean_mfcc('valid')
    test_data = mean_mfcc('test')

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.show()








