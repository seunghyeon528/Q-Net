# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import time
import librosa
import math
import pdb



####################################################################
#                             Prepare Data
####################################################################
# train_list �� ���� 15000 �� list�̸�
# �� element�� clean ���� �ּ�, dedgraded���� �ּ�, PESQ ���� value�� ���� dictionary ��.
# ���� clean�� �Ǵ� ���ϸ����� SNR �Ǵ� Noise�� �ܾ ����.
def prepare_data(label_path):
    f = open(label_path)
    lines = f.readlines()
    file_list = list()
    
    for i in range(len(lines)):
        temp_dic = {'clean':'' ,'degraded':'', 'PESQ':0}
        temp_slice = lines[i].split("\t")
        temp_path = temp_slice[0]
        temp_PESQ = float(temp_slice[1].strip())
        temp_PESQ = temp_PESQ/4.5
        if temp_PESQ < 0:
            temp_PESQ = 0

        if "enh" in lines[i]: # the file belongs to nenw 10,000 dataset
                temp_dic['degraded'] = temp_path
                temp_dic['PESQ'] = temp_PESQ
                temp_clean =  (temp_path.replace("/enhance","/target",1)).replace("enh_","tar_") # it should be modified
                temp_dic['clean'] = temp_clean
                file_list.append(temp_dic)            
        else:
            if "SNR" in lines[i]:
                temp_dic['degraded'] = temp_path
                temp_dic['PESQ'] = temp_PESQ
                temp_clean = temp_slice[0].split("_SNR_")[0] + ".WAV"
                temp_dic['clean'] = temp_clean
                file_list.append(temp_dic)
            else:
                temp_dic['degraded'] = temp_path
                temp_dic['PESQ'] = temp_PESQ
                temp_dic['clean'] = temp_path
                file_list.append(temp_dic)

    f.close()
    return file_list

####################################################################
#                             Prepare Data
###################################################################
class Mydataset2(Dataset):
    def __init__(self,filelist):
        self.filelist = filelist
        self.n_fft = int(512)
        self.hop_length = int(self.n_fft/4)


    def __len__(self):
        return len(self.filelist)


    def prepare_sample(self, waveform):
        wave_len = waveform.shape[0]
        output = np.zeros(self.maxlen)
        output[:wave_len] = waveform
        return output

    def __getitem__(self,index):
        
        clean_file = self.filelist[index]['clean']
        degraded_file = self.filelist[index]['degraded']
        PESQ = self.filelist[index]['PESQ']
        
        # clean, degraded wav file�κ��� ���� stft magnitude ����
        eps = 1e-8
        clean_wav, fs = librosa.load(clean_file,sr = 16000)
        clean_spec_wav = librosa.stft(clean_wav,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,center=False)
        clean_magnitude_spec =np.abs(clean_spec_wav) # magnitude

        degraded_wav, fs = librosa.load(degraded_file, sr = 16000)
        degradeed_spec_wav = librosa.stft(degraded_wav,n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.n_fft,center=False)
        degraded_magnitude_spec =np.abs(degradeed_spec_wav)

        # clean, degraded concatenate
        m_stft = np.stack([clean_magnitude_spec, degraded_magnitude_spec], axis=0)
        # m_stft = torch.stack([torch.Tensor(clean_log_spec),torch.Tensor(degraded_log_spec)],dim = 0)
        return m_stft,PESQ


def my_collate1(batch):
    m_stft, PESQ = list(zip(*batch))
    m_stft = torch.FloatTensor(m_stft)
    PESQ = torch.FloatTensor(PESQ)
    return (m_stft, PESQ)

def my_collate(batch):
   
    m_stft, PESQ = list(zip(*batch))
    data_len = torch.LongTensor(np.array([x.shape[2] for x in m_stft]))
    max_len = max(data_len)
    B = len(m_stft)
    inputs = torch.zeros(B,2, 257, max_len)
    

    for i in range(B):
        inputs[i,:, :, :m_stft[i].shape[2]] = torch.Tensor(m_stft[i])

    PESQ = torch.FloatTensor(PESQ)
    return (inputs, PESQ)

