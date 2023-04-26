# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
from pydub import AudioSegment

def clip_test(dir):
    """分离训练集测试集"""
    txt_dir = dir+"ICBHI_challenge_train_test.txt"
    
    with open(txt_dir,'r') as f:
        name = []
        set_type = []
        for row in f.readlines():
            row = row.strip('\n')
            row = row.split('\t')
            
            name.append(row[0])
            set_type.append(row[1])
    
    for i in range(len(name)):
        '''
        这里将测试集移动出去，就是将整个集合进行分离
        '''
        if set_type[i]=='test':
            shutil.move(dir+'train_set/'+name[i]+'.wav', dir+'test_set/'+name[i]+'.wav') 
            
def clip_cycle(dir,new_dir):
    """
    将wav文件分割
    dir : trainset/testset record path
    new_dir:breath cycle save path
    """
    for file in os.listdir(dir):
        txt_name = '../data/ICBHI/'+file[:-4]+'.txt'
        time = np.loadtxt(txt_name)[:,0:2]
        sound = AudioSegment.from_wav(dir+file)
        for i in range(time.shape[0]):
            start_time = time[i,0]*1000
            stop_time = time[i,1]*1000
            word = sound[start_time:stop_time]
            word.export(new_dir+file[:-4]+str(i)+'.wav', format="wav")

clip_test("../data/")
clip_cycle("../data/train_set/","../data/train/")
clip_cycle("../data/test_set/","../data/test/")
