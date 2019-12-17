#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
#np.random.seed(1969)
import tensorflow as tf

from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank


# In[2]:


L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
train_audio_path = '../data/train/audio'


# # File names processing

# In[3]:


def list_wav_fnames(dirpath, batch_size=0):
    '''
    get wav files from desired directories
    
    dirpath: path storing all audio data
    batch_size: pick how many files from one directory
        if batch_size = 0, get all the files from one directory
    '''
    train_labels = os.listdir(dirpath)
    if '.DS_Store' in train_labels:
        train_labels.remove('.DS_Store')

    train_file_labels = []
    for label in train_labels:
        files = np.array(os.listdir(dirpath + '/' + label))
        #print(len(files))
        if batch_size < len(files) and batch_size > 0:
            chosen_indices = []
            
            for i in range(batch_size):
                index = np.random.randint(0, len(files))
                while index in chosen_indices:
                    index = np.random.randint(0, len(files))
                    
                chosen_indices.append(index)
        
            files = files[chosen_indices]
        for f in files:

            if f.endswith('.wav'):
                train_file_labels.append((label, f))
            
    return train_file_labels


# # Processing Waves

# In[4]:


def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=1000):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]








