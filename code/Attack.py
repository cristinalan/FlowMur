import random
import csv
import numpy as np
import torchaudio.transforms as T
from parameters import *
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from utils import load_model,train,test,get_data_label,EarlyStoppingModel
from model import smallcnn,largecnn,smalllstm,RNN,ResNet,ResidualBlock
from sklearn.model_selection import train_test_split


mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            #"n_mels": n_mels,
            "hop_length": HOP_LENGTH,
            #"mel_scale": "htk",
        },)

#FKD: 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000, 1.125, 1.25
#SCD: 0.1    0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,   1
# Trigger_Duration = 0.5
# kind = "SCD-5"
# Target_Label = 5

for i in [0.7]:
#for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]: #poisoning rate
    print("---------------------",i,"-------------------")

    model = largecnn().to(DEVICE)
    #model = smallcnn().to(DEVICE)
    #model = RNN().to(DEVICE)
    #model = ResNet(ResidualBlock, [2, 2, 2]).to(DEVICE)
    trigger_path = "./project/rnn/sp_trigger1000.npy"
    #trigger_path = "./FKD/SmallLSTM-trigger/trigger_0.625_1000.npy"
    dataset = "sp"  #please select "football" or 'sp'
    if dataset == "football":
        train_waveform = torch.tensor(np.load("./Dataset/FKD_train_waveform_8.npy"))
        train_label = torch.tensor(np.load("./Dataset/FKD_train_label_8.npy")).long()
        train_waveform, validation_waveform, train_label,validation_label = train_test_split(train_waveform, train_label, test_size=0.2,random_state=1)
        #print(validation_label)
        test_waveform = torch.tensor(np.load("./Dataset/FKD_test_waveform_8.npy")) #<class 'numpy.ndarray'>   (35628, 1, 16000)
        test_label = torch.tensor(np.load("./Dataset/FKD_test_label_8.npy")).long()

    elif dataset == "sp":
        train_waveform = torch.tensor(np.load("./Dataset/SCD_train_waveform_10.npy"))
        train_label = torch.tensor(np.load("./Dataset/SCD_train_label_10.npy")).long()
        train_waveform, validation_waveform, train_label, validation_label = train_test_split(train_waveform,train_label,test_size=0.2,random_state=1)
        test_waveform = torch.tensor(np.load("./Dataset/SCD_test_waveform_10.npy"))  # <class 'numpy.ndarray'>   (35628, 1, 16000)
        test_label = torch.tensor(np.load("./Dataset/SCD_test_label_10.npy")).long()

    test_waveform_no = test_waveform.clone()
    test_label_no = test_label.clone()
    target_class_index = np.where(train_label == Target_Label)[0]  # <class 'numpy.ndarray'>
    print(target_class_index.shape)
    #random trigger
    #trigger = torch.tensor(-0.2 + 0.4*np.random.random((1,8000))).to(DEVICE)
    #0.1trigger
    #trigger = 0.1*torch.ones((1,8000)).to(DEVICE)
    #optimized trigger
    #trigger = torch.tensor(np.load(trigger_path)).to(DEVICE)
    trigger = torch.tensor(np.load(trigger_path))
    #trigger = torch.zeros(1,16000)
    print(trigger.shape)
    print(trigger)
    print("trigger.device:",trigger.device)

    #构造投毒训练集
    poison_num = int(target_class_index.shape[0] * i)
    print("poison num:",poison_num)
    poison_index = np.random.choice(target_class_index,poison_num,replace=False)
    print(poison_index.shape)
    trigger_rms = torch.linalg.norm(trigger.clone(),dim=1)
    print("trigger_rms:",trigger_rms)
    for i in poison_index:
        wav_rms = torch.linalg.norm(train_waveform[i].clone(), dim=1)
        scale = torch.sqrt(torch.pow(wav_rms, 2) / torch.pow(trigger_rms, 2) * (10 ** (-Snr_DB / 10)))
        position = random.randint(0, train_waveform.shape[2] - trigger.shape[1])
        befo_tr = train_waveform[i][0][0:position]
        in_tr = train_waveform[i][0][position:position + trigger.shape[1]] + scale * trigger[0]
        af_tr = train_waveform[i][0][position + trigger.shape[1]:]
        train_waveform[i] = torch.cat([befo_tr, in_tr, af_tr]).unsqueeze(dim=0)
    print(train_waveform.shape)
    print(train_label.shape)
    train_label = train_label.to(DEVICE)
    train_mfcc = mfcc_transform(train_waveform).permute(0,1,3,2).to(DEVICE)
    print(train_mfcc.shape)
    print("mfcc transform is ended")
    train_poisoned_dataset = Data.TensorDataset(train_mfcc, train_label)
    train_poisoned_loader = Data.DataLoader(train_poisoned_dataset,BATCH_SIZE,shuffle=True)


    #构造良性测试集
    test_mfcc = mfcc_transform(test_waveform).permute(0,1,3,2).to(DEVICE)
    test_label = test_label.to(DEVICE)
    test_benign_dataset = Data.TensorDataset(test_mfcc,test_label)
    test_benign_loader = Data.DataLoader(test_benign_dataset,BATCH_SIZE,shuffle=True)
    #构造投毒测试集
    #test_poisoned_waveform = test_waveform.to(DEVICE)+trigger

    print("test_waveform.shape_before:",test_waveform_no.shape)
    print(type(test_label))
    target_class_index_test = np.where(test_label_no == Target_Label)[0]
    print("target_class_index_test.shape:",target_class_index_test.shape)
    test_waveform_no = np.delete(test_waveform_no,target_class_index_test,axis=0)
    print("test_waveform.shape_after:", test_waveform_no.shape)

    for i in range(test_waveform_no.shape[0]):
        position = random.randint(0,test_waveform_no.shape[2]-trigger.shape[1])
        befo_tr = test_waveform_no[i][0][0:position]/2
        in_tr = (test_waveform_no[i][0][position:position+trigger.shape[1]]+trigger[0])/2
        af_tr = test_waveform_no[i][0][position+trigger.shape[1]:]/2
        test_waveform_no[i] = torch.cat([befo_tr,in_tr,af_tr]).unsqueeze(dim=0)
    test_poisoned_mfcc = mfcc_transform(test_waveform_no).permute(0,1,3,2).to(DEVICE)
    test_poisoned_label = torch.tensor([Target_Label]*test_waveform_no.shape[0]).to(DEVICE)
    test_poisoned_dataset = Data.TensorDataset(test_poisoned_mfcc,test_poisoned_label)
    test_poisoned_loader = Data.DataLoader(test_poisoned_dataset,BATCH_SIZE,shuffle=True)
    #构造验证集
    #validation_waveform = validation_waveform
    validation_label = validation_label.to(DEVICE)
    validation_mfcc = mfcc_transform(validation_waveform).permute(0,1,3,2).to(DEVICE)
    #训练poisoned model
    optimizer = optim.Adam(model.parameters(),lr=Learning_Rate_Poisoned_Model)


    #poisoned_model_path = str(Trigger_Duration)+"_"+str(Snr_DB)+"_"+str(poison_num)+"_poisoned_model.pkl"
    poisoned_model_path = "./project/poisonous_model/SCD_smallcnn_largecnn.pkl"
    early_stopping = EarlyStoppingModel(patience=Patience_Poisoned_Model, verbose=True,path=poisoned_model_path)
    for epoch in range(1,EPOCH+1):
        print("----- Epoch ", epoch, " -----")
        train(model,train_poisoned_loader,optimizer)
        model.eval()
        valid_output = model(validation_mfcc)
        valid_loss = F.nll_loss(valid_output.squeeze(), validation_label)
        early_stopping(valid_loss, model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break

    print(trigger_path)
    print(Snr_DB)
    print("poison num:",poison_num)
    poisoned_model = load_model(poisoned_model_path,DEVICE)
    #benign_model = load_model("TEST_model.pkl",DEVICE)
    BA = test(poisoned_model,test_benign_loader)
    print("BA:",BA)
    ASR = test(poisoned_model,test_poisoned_loader)
    print("ASR:",ASR)

    with open("./project/smallcnn-largecnn-SCD.csv",'a') as csvfile:
        writer = csv.writer(csvfile,lineterminator="\n")
        writer.writerow([])
        writer.writerow([Snr_DB,poison_num,epoch,BA,ASR])