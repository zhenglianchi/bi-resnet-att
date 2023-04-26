# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import math
import torch
import librosa.display as display

def save_pic(wav_dir,save_dir):
    txt_name = ''
    txt_dir ='../data/ICBHI/'
    for file in os.listdir(wav_dir):
        num = file[-5]
        if file[:22]!=txt_name[:-4]:
            txt_name = file[:22]+'.txt'
            array = np.loadtxt(txt_dir+txt_name)
            label = array[:,2:4]
        
        #这里如果采用librosa会自动进行归一化
        
        #fs,sig= wav.read(wav_dir+'/'+file)
        sig,fs= librosa.load(wav_dir+'/'+file,sr=None)

        if fs>22050:
            sig = librosa.resample(y=sig,orig_sr=fs,target_sr=22050)

        targetsample=22050*8
        ratio = math.ceil(targetsample / sig.shape[-1])
        sig = torch.tensor(sig).repeat(ratio)
        sig = sig[...,:targetsample].numpy()
        
        melspec=librosa.feature.melspectrogram(y=sig,sr=fs,win_length=2048,hop_length=512)

        display.specshow(librosa.amplitude_to_db(melspec,ref=np.max),y_axis='log',x_axis='time')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)

        crackles = label[int(num),0]
        wheezes = label[int(num),1]
        if crackles==0 and wheezes==0:
            plt.savefig(save_dir+'zero/'+file[:24]+'png', cmap='Greys_r')
        elif crackles==1 and wheezes==0:
            plt.savefig(save_dir+'one/'+file[:24]+'png', cmap='Greys_r')
        elif crackles==0 and wheezes==1:
            plt.savefig(save_dir+'two/'+file[:24]+'png', cmap='Greys_r')
        else:
            plt.savefig(save_dir+'three/'+file[:24]+'png', cmap='Greys_r')
        plt.clf()
        
        
if __name__ == '__main__':
   save_pic('../data/train','../analysis/melspec/train/')
   save_pic('../data/test','../analysis/melspec/test/')
