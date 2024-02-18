import torch.utils.data as Data
from utils import load_model, train, test, get_data_label
import numpy as np
import random
import torchaudio
from parameters import *
from utils import deploy_trigger_to_waveform,EarlyStoppingTrigger
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

def generate_trigger(benign_model,dataloader,Trigger_Length):
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            # "n_mels": n_mels,
            "hop_length": HOP_LENGTH,
            #"mel_scale": "htk",
        },
    ).to(DEVICE)

    for param in benign_model.parameters():
        param.requires_grad = False


    #初始化触发器的值

    #trigger_initial = torch.tensor(-0.1+0.2*np.random.random((1,20000))).float().to(DEVICE)
    #trigger_initial = trigger_init.float().to(DEVICE)
    trigger_initial = torch.ones((1,Trigger_Length),device=DEVICE)*Trigger_Initial_Value
    # templete,_ = torchaudio.load("rain_1_25.wav")
    # templete = templete.to(DEVICE)
    trigger = torch.autograd.Variable(trigger_initial,requires_grad=True)
    print("initial trigger:",trigger)
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(params=[trigger], lr=Learning_Rate_Generate_Trigger)
    optimizer = torch.optim.Adam(params=[trigger], lr=0.001)

    #early_stopping = EarlyStoppingTrigger(Patience_Generate_Trigger, verbose=True,path=path)
    torch.backends.cudnn.enabled = False
    for epoch in range(1,EPOCH+1):
        print("----- Epoch ", epoch, " -----")
        loss = 0
        for waveforms,labels in dataloader:

            new_waveforms = deploy_trigger_to_waveform(waveforms,trigger)
            waveforms = torch.clamp(new_waveforms,-1,1)
            mfccs = mfcc_transform(waveforms).permute(0,1,3,2).squeeze(dim=1)
            #print(mfccs.shape)
            pred = benign_model.forward(mfccs)
            trigger_now = trigger.data
            #loss = loss + criterion(pred,labels)+0.1*torch.norm((trigger-templete).float())+0.1*torch.norm(trigger_now)
            loss = loss + criterion(pred,labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            trigger.data = torch.clamp(trigger.data,-Epsilon,Epsilon)

        trigger_save = trigger.cpu().clone()
        trigger_data = torch.autograd.Variable(trigger_save, requires_grad=False).numpy()

        if epoch%200 == 0:
            path = "./project/largecnn/sp_trigger" + str(epoch) + ".npy"
            np.save(path,trigger_data)

        print(loss)
        '''early_stopping(loss, trigger_data)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break'''

    print("last trigger:", trigger)

    #np.save(path, trigger_data)
    return torch.autograd.Variable(trigger, requires_grad=False)


#for Trigger_Duration in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
for Trigger_Duration in [0.5]:
    Trigger_Length = int(Trigger_Duration * SAMPLE_RATE)
    dataset = "sp"  #please select "football" or 'sp'

    if dataset == "football":
        train_waveform = torch.tensor(np.load("./Dataset/FKD_train_waveform_11.npy"))
        train_label = torch.tensor(np.load("./Dataset/FKD_train_label_11.npy")).long()
        train_waveform, validation_waveform, train_label,validation_label = train_test_split(train_waveform, train_label, test_size=0.2,random_state=1)
        #print(validation_label)
        test_waveform = torch.tensor(np.load("./Dataset/FKD_test_waveform_11.npy")) #<class 'numpy.ndarray'>   (35628, 1, 16000)
        test_label = torch.tensor(np.load("./Dataset/FKD_test_label_11.npy")).long()

    elif dataset == "sp":
        train_waveform = torch.tensor(np.load("./Dataset/SCD_train_waveform_26.npy"))
        train_label = torch.tensor(np.load("./Dataset/SCD_train_label_26.npy")).long()
        train_waveform, validation_waveform, train_label, validation_label = train_test_split(train_waveform, train_label,
                                                                                      test_size=0.2, random_state=1)
        # print(validation_label)
        test_waveform = torch.tensor(np.load("./Dataset/SCD_test_waveform_26.npy"))  # <class 'numpy.ndarray'>   (35628, 1, 16000)
        test_label = torch.tensor(np.load("./Dataset/SCD_test_label_26.npy")).long()
    '''
    #提取目标类训练集，并构造dataloader
    target_class_index = np.where(train_label == Target_Label)[0]  #<class 'numpy.ndarray'>
    print(target_class_index.shape)
    train_waveform_target_class = train_waveform[target_class_index]    #torch.Size([3011, 1, 16000])
    train_label_target_class = train_label[target_class_index]     #torch.Size([3011])
    train_target_class_dataset = Data.TensorDataset(train_waveform_target_class.to(DEVICE), train_label_target_class.to(DEVICE))
    train_target_class_dataloader = Data.DataLoader(train_target_class_dataset,BATCH_SIZE,shuffle=True)
    '''
    print(train_waveform.shape)
    index = random.sample(range(train_waveform.shape[0]),5000)
    train_waveform_use = train_waveform[index]
    print(train_waveform_use.shape)
    train_label = torch.tensor([Target_Label]*5000).to(DEVICE)
    train_dataset = Data.TensorDataset(train_waveform_use.to(DEVICE),train_label.to(DEVICE))
    train_dataloader = Data.DataLoader(train_dataset,BATCH_SIZE,shuffle=True)

    benign_model = load_model("./project/smallcnn/sp_smallcnn_26.pkl", DEVICE)
    #benign_model = load_model("./FKD/SmallLSTM-trigger/FKD_smalllstm_11.pkl", DEVICE)
    trigger = generate_trigger(benign_model,train_dataloader,Trigger_Length)

    print(trigger.shape)
    print("The trigger has been generated!")
    print(trigger)