
import torch
from parameters import *
import torch.optim as optim
import torch.nn.functional as F
from utils import get_data_label,train,test,save_model,load_model,EarlyStoppingModel
import torch.utils.data as Data
from model import smallcnn,largecnn,smalllstm,RNN,ResNet,ResidualBlock
import numpy as np
from sklearn.model_selection import train_test_split

dataset = "sp"  #please select "football" or 'sp'
kind = ""

if dataset == "football":
    train_data = torch.tensor(np.load("./Dataset/FKD_train_mfcc_8.npy"))
    #print(train_data.shape)
    train_label = torch.tensor(np.load("./Dataset/FKD_train_label_8.npy")).long()
    train_data, validation_data, train_label,validation_label =  train_test_split(train_data, train_label, test_size=0.2,random_state=1)
    #print(validation_label)
    test_data = torch.tensor(np.load("./Dataset/FKD_test_mfcc_8.npy"))
    test_label = torch.tensor(np.load("./Dataset/FKD_test_label_8.npy")).long()

elif dataset == "sp":
    train_data = torch.tensor(np.load("./Dataset/SCD_train_mfcc_10.npy"))
    # print(train_data.shape
    train_label = torch.tensor(np.load("./Dataset/SCD_train_label_10.npy")).long()
    train_data, validation_data, train_label, validation_label = train_test_split(train_data, train_label,
                                                                                  test_size=0.2, random_state=1)
    # print(validation_label)
    test_data = torch.tensor(np.load("./Dataset/SCD_test_mfcc_10.npy"))
    test_label = torch.tensor(np.load("./Dataset/SCD_test_label_10.npy")).long()


train_data, validation_data, test_data, train_label, validation_label, test_label = train_data.to(DEVICE), validation_data.to(DEVICE), test_data.to(DEVICE), train_label.to(DEVICE), validation_label.to(DEVICE), test_label.to(DEVICE)
train_dataset = Data.TensorDataset(train_data, train_label)
validation_dataset = Data.TensorDataset(validation_data, validation_label)
test_dataset = Data.TensorDataset(test_data, test_label)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = Data.DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = torch.nn.CrossEntropyLoss().cuda()

epoches = []
accs = []
for i in range(5):
    model = smallcnn().to(DEVICE)
    #model = largecnn().to(DEVICE)
    #model = smalllstm().to(DEVICE)
    #model = RNN().to(DEVICE)
    #model = ResNet(ResidualBlock, [2, 2, 2]).to(DEVICE)   #11559,1,40,13

    optimizer = optim.Adam(model.parameters(),lr=Learning_Rate_Benign_Model)
    path = "./project/smallcnn/"+dataset+"_smallcnn_10_"+str(i)+".pkl"
    print("path",path)
    early_stopping = EarlyStoppingModel(patience=Patience_Benign_Model,verbose=True,path=path)
    print("learning rate:",Learning_Rate_Benign_Model)
    print("patience:",Patience_Benign_Model)

    for epoch in range(1,EPOCH+1):
      print("----- Epoch ",epoch," -----")
      train(model,train_loader,optimizer)
      model.eval()
      valid_output = model(validation_data)
      #valid_loss = F.nll_loss(valid_output.squeeze(), validation_label)
      valid_loss = criterion(valid_output.squeeze(),validation_label)
      early_stopping(valid_loss, model)
      # 若满足 early stopping 要求
      if early_stopping.early_stop:
          print("Early stopping")
          # 结束模型训练
          break
      torch.cuda.empty_cache()
    epoches.append(epoch)
    print("epoches",epoches)

    benign_model = load_model(path,DEVICE)
    acc = test(benign_model,test_loader)
    accs.append(acc)
    print("accs",accs)