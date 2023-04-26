# -*- coding: utf-8 -*-
import joblib
import os
from PIL import Image
import numpy as np


def pack(dir_mfccs,dir_melspec,label):       
    feature_mfccs_list=[]
    feature_melspec_list=[]
    label_list=[] 
    for file in os.listdir(dir_mfccs):
        I_mfccs = Image.open(dir_mfccs+file).convert('L')
        I_melspec = Image.open(dir_melspec+file).convert('L')
        I_mfccs = np.array(I_mfccs)
        I_melspec = np.array(I_melspec)

        feature_melspec_list.append(I_melspec)
        feature_mfccs_list.append(I_mfccs)
        label_list.append(label)
    return feature_mfccs_list,feature_melspec_list,label_list
    

    
if __name__ == '__main__':
    mfccs0,melspec0,label0 = pack('../analysis/mfccs/test/zero/','../analysis/melspec/test/zero/',0)
    mfccs1,melspec1,label1 = pack('../analysis/mfccs/test/one/','../analysis/melspec/test/one/',1)
    mfccs2,melspec2,label2 = pack('../analysis/mfccs/test/two/','../analysis/melspec/test/two/',2)
    mfccs3,melspec3,label3 = pack('../analysis/mfccs/test/three/','../analysis/melspec/test/three/',3)
    mfccs = mfccs0+mfccs1+mfccs2+mfccs3
    melspec = melspec0+melspec1+melspec2+melspec3
    label = label0+label1+label2+label3
    #joblib.dump((mfccs,melspec, label), open('../analysis/pack/melspec_mfccs_train.p', 'wb'))
    joblib.dump((mfccs,melspec,label), open('../analysis/pack/melspec_mfccs_test.p', 'wb'))