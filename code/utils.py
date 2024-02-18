import random
import numpy as np
from parameters import *
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

def get_data_label(data_path,label_path):

    total_data = torch.tensor(np.load(data_path,allow_pickle=True)) #<class 'numpy.ndarray'>   (35628, 1, 16000)
    total_label = torch.tensor(np.load(label_path,allow_pickle=True)).long()   #<class 'numpy.ndarray'>   (35628,)

    train_data, test_data, train_label, test_label = train_test_split(total_data, total_label, test_size=0.2,
                                                                      random_state=1)
    train_data, validation_data, train_label,validation_label =  train_test_split(train_data, train_label, test_size=0.2,
                                                                      random_state=1)
    #train_dataset = Data.TensorDataset(train_waveform, train_label)
    #test_dataset = Data.TensorDataset(test_waveform, test_label)
    #train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_data, validation_data, test_data, train_label, validation_label, test_label


def test(model,test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

    print(f"\nAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.3f}%)\n")
    return round(correct / len(test_loader.dataset),4)

def train(model, train_loader, optimizer):

    model.train()
    losses = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(DEVICE)
        target = target.to(DEVICE)
        #print("batch_train_data:", data.shape)

        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        #loss = F.nll_loss(output.squeeze(), target)
        loss = criterion(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % LOG_INTERVAL == 0:
            print(f" [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        losses.append(loss.item())

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def save_model(net,path):
    torch.save(net,path)

def load_model(path,map_location):
    net = torch.load(path,map_location)
    return net

def deploy_trigger_to_waveform(waveforms,trigger):
    waveforms_rms = torch.linalg.norm(waveforms,dim=2)
    trigger_rms = torch.linalg.norm(trigger.clone(),dim=1)
    scale = 10**(Snr_DB/20)*(trigger_rms/waveforms_rms)
    new_waveforms = torch.tensor([]).to(DEVICE)
    for i,wav in enumerate(waveforms):
        position = random.randint(0, waveforms.shape[2] - trigger.shape[1])
        #position = 0
        befo_tr = scale[i]*wav[0][0:position]/(scale[i]+1)
        in_tr = (scale[i]*wav[0][position:position+trigger.shape[1]]+trigger[0])/(scale[i]+1)
        af_tr = scale[i]*wav[0][position + trigger.shape[1]:]/(scale[i]+1)
        new_wav = torch.cat([befo_tr,in_tr,af_tr]).unsqueeze(dim=0).unsqueeze(dim=0)
        new_waveforms = torch.cat((new_waveforms,new_wav),dim=0)
    return new_waveforms



class EarlyStoppingModel:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        save_model(model, self.path)

        self.val_loss_min = val_loss
        
        
        
class EarlyStoppingTrigger:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, trigger_data):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,trigger_data)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, trigger_data)
            self.counter = 0

    def save_checkpoint(self, val_loss, trigger_data):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving trigger ...')

        np.save(self.path,trigger_data)

        self.val_loss_min = val_loss
