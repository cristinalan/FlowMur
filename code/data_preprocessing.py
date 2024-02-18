import os
import torch
import torchaudio
from parameters import *
import numpy as np
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

def MFCC(waveform):
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            #"n_mels": n_mels,
            "hop_length": HOP_LENGTH,
            #"mel_scale": "htk",
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

def prepare_data_sp(data_path,labels,waveform_to_consider):
    #data_path
    #labels
    #waveform_to_consider
    total_waveform = []
    total_label = []
    total_mfcc = []
    for label in labels:
        label_path = os.path.join(data_path,label)
        print(label_path)
        wav_names = os.listdir(label_path)
        print(wav_names[:10])
        for wav_name in wav_names:
            if wav_name.endswith(".wav"):
                wav_path = os.path.join(label_path,wav_name)
                waveform, sample_rate = torchaudio.load(wav_path)
                if waveform.shape[1] >= waveform_to_consider:
                    waveform = waveform[:waveform_to_consider]
                    total_waveform.append(waveform.numpy())
                    total_label.append(torch.tensor(labels.index(label)))
                    #print(MFCC(waveform.squeeze(0)).shape)
                    total_mfcc.append(MFCC(waveform.squeeze(0)).numpy().T[np.newaxis,:])

    train_waveform,test_waveform,train_mfcc,test_mfcc,train_label,test_label = \
        train_test_split(total_waveform, total_mfcc,total_label,test_size=0.2, random_state=1)
    train_waveform = np.array(train_waveform)
    test_waveform = np.array(test_waveform)
    train_mfcc = np.array(train_mfcc)
    test_mfcc = np.array(test_mfcc)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    np.save("./Dataset/SCD_train_waveform_10", train_waveform)
    np.save("./Dataset/SCD_test_waveform_10", test_waveform)
    np.save("./Dataset/SCD_train_mfcc_10", train_mfcc)
    np.save("./Dataset/SCD_test_mfcc_10", test_mfcc)
    np.save("./Dataset/SCD_train_label_10", train_label)
    np.save("./Dataset/SCD_test_label_10", test_label)


def prepare_data_football(data_train_path,data_test_path,labels_football):
    train_waveform = []
    train_label = []
    train_mfcc = []
    test_waveform = []
    test_label = []
    test_mfcc = []
    train_wav_names = os.listdir(data_train_path)
    for train_wav_name in train_wav_names:
        for label in labels_football:
            if label in train_wav_name:
                wav_path = os.path.join(data_train_path,train_wav_name)
                waveform,sample_rate = torchaudio.load(wav_path)
                train_waveform.append(waveform.numpy())
                train_label.append(torch.tensor(labels_football.index(label)))
                train_mfcc.append(MFCC(waveform).numpy().T[np.newaxis, :])
    train_waveform = np.array(train_waveform)
    train_label = np.array(train_label)
    print(train_waveform.shape)
    print(train_label.shape)
    #print(np.array(train_mfcc).shape)
    np.save("/root/autodl-tmp/project/Dataset/FKD_train_waveform_11.npy", train_waveform)
    np.save("/root/autodl-tmp/project/Dataset/FKD_train_label_11.npy", train_label)
    np.save("/root/autodl-tmp/project/Dataset/FKD_train_mfcc_11.npy", train_mfcc)

    test_wav_names = os.listdir(data_test_path)
    for test_wav_name in test_wav_names:
        for label in labels_football:
            if label in test_wav_name:
                wav_path = os.path.join(data_test_path, test_wav_name)
                waveform, sample_rate = torchaudio.load(wav_path)
                test_waveform.append(waveform.numpy())
                test_label.append(torch.tensor(labels_football.index(label)))
                test_mfcc.append(MFCC(waveform).squeeze(0).numpy().T[np.newaxis, :])
    test_waveform = np.array(test_waveform)
    test_label = np.array(test_label)
    print(test_waveform.shape)
    print(test_label.shape)
    # print(np.array(test_mfcc).shape)
    
    # np.save("./Dataset/FKD-K00008/football_test_waveform_11.npy", test_waveform)
    # np.save("./Dataset/FKD-K00008/football_test_label_11.npy", test_label)
    # np.save("./Dataset/FKD-K00008/football_test_mfcc_11.npy", test_mfcc)
    np.save("/root/autodl-tmp/project/Dataset/FKD_test_waveform_11.npy", test_waveform)
    np.save("/root/autodl-tmp/project/Dataset/FKD_test_label_11.npy", test_label)
    np.save("/root/autodl-tmp/project/Dataset/FKD_test_mfcc_11.npy", test_mfcc)

data_path = "./dataset/SpeechCommands"

data_train_path = './dataset/FootballKeywords/train'
data_test_path = './dataset/FootballKeywords/test (single keyword)'
#data_test_path = '/Users/lan/Documents/Code/dataset/Football Keywords Dataset V1.0/test (single keyword)'
labels_26 = ["zero","backward","bed","bird","cat","dog","down","follow","forward","go","happy","house","learn","left","marvin","no","off","on","right","sheila","stop","tree","up","visual","wow","yes"]
labels_10 = ["zero","one","two","three","four","five","six","seven","eight","nine"]

#labels_sp = ["zero","one","two","three","four","five","six","seven","eight","nine"]
labels_football_8 = ['K00001','K00002','K00003','K00004','K00005','K00006','K00007','K00008']
labels_football_11 = ['K00001','K00009','K00010','K00011','K00012','K00013','K00014','K00015','K00016','K00017','K00018']

waveform_to_consider = 16000
prepare_data_sp(data_path,labels_10,waveform_to_consider)
# prepare_data_football(data_train_path,data_test_path,labels_football_11)