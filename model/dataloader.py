import numpy as np
import torch
import joblib
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class myDataset(data.Dataset):
    def __init__(self, mfccs, melspec, targets):
        self.mfccs = mfccs
        self.melspec = melspec
        self.targets = targets

    def __getitem__(self, index):
        sample_mfccs = self.mfccs[index]
        sample_melspec = self.melspec[index]
        target = self.targets[index]
        min_s = np.min(sample_mfccs)
        max_s = np.max(sample_mfccs)
        sample_mfccs = (sample_mfccs-min_s)/(max_s-min_s) 
        min_m = np.min(sample_melspec)
        max_m = np.max(sample_melspec)
        sample_melspec = (sample_melspec-min_m)/(max_m-min_m) 
        
        output_mfccs = torch.FloatTensor(np.array([sample_mfccs]))
        crop_s = transforms.Resize([128,128])
        img_s = transforms.ToPILImage()(output_mfccs)
        croped_img=crop_s(img_s)
        output_mfccs = transforms.ToTensor()(croped_img)
        
        output_melspec = torch.FloatTensor(np.array([sample_melspec]))
        crop_m = transforms.Resize([128,128])
        img_m = transforms.ToPILImage()(output_melspec)
        croped_img_m=crop_m(img_m)
        output_melspec = transforms.ToTensor()(croped_img_m)

        return output_mfccs,output_melspec,target
    def __len__(self):
        return len(self.targets)


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000,use_k=False):
    mfccs, melspec, labels = joblib.load(open('../analysis/pack/melspec_mfccs_train.p', mode='rb'))
    mfccs_test, melspec_test, labels_test = joblib.load(open('../analysis/pack/melspec_mfccs_test.p', mode='rb'))

    if use_k:
        mfccs_=mfccs+mfccs_test
        melspec_=melspec+melspec_test
        labels_=labels+labels_test
        return np.array(mfccs_),np.array(melspec_),np.array(labels_)
        
    train_loader = DataLoader(
        myDataset(mfccs, melspec, labels), batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )

    test_loader = DataLoader(
        myDataset(mfccs_test, melspec_test, labels_test),
        batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    return train_loader, test_loader