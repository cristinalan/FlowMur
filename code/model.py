import torch
from parameters import *
import torch.nn as nn
import torch.nn.functional as F

#smallcnn
#class model_adv_detetcion(nn.Module):
class smallcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(2,2))
        self.bn1 = nn.BatchNorm2d(num_features=64)  #即channel的数字
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(2,2))
        self.bn2 = nn.BatchNorm2d(num_features=64)  #即channel的数字
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),padding=(1,1))

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(2,2))
        self.bn3 = nn.BatchNorm2d(num_features=32)  #即channel的数字
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),padding=(0,1))
        self.drop1 = nn.Dropout(0.4)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(in_features=224,out_features=128)  #256,1,32,13 SPEECHCOMMAND
        #self.fc1 = nn.Linear(in_features=288, out_features=128)  # 256,1,40,13 FOOTBALL
        self.drop2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=128,out_features=10)
        self.softmax = nn.Softmax()
        # torch.cuda.manual_seed(42)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop1(x)

        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

#largecnn
class largecnn(nn.Module):
#class model_trojaning_attacks(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=(3,3),stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))

        self.flat = nn.Flatten()
        #self.fc1 = nn.Linear(in_features=1024, out_features=256)   #256，1，40,13,football
        self.fc1 = nn.Linear(in_features=768,out_features=256)      #sample_rate=16000，256，1，32，13

        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=128,out_features=10)
        # self.softmax = F.log_softmax(dim=1)
        #torch.cuda.manual_seed(42)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        #output = F.softmax(x, dim=1)
        return output

class smalllstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(2,2))
        self.bn1 = nn.BatchNorm2d(num_features=64)  #即channel的数字
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(2,2))
        self.bn2 = nn.BatchNorm2d(num_features=64)  #即channel的数字
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),padding=(1,1))

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(2,2))
        self.bn3 = nn.BatchNorm2d(num_features=32)  #即channel的数字
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),padding=(0,1))
        self.drop1 = nn.Dropout(0.4)

        self.flat = nn.Flatten()
        self.rnn = nn.LSTM(32, 128, 2, batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(in_features=224,out_features=128)  #256,1,32,13
        #self.fc1 = nn.Linear(in_features=672, out_features=128)  # 256,1,87,13
        self.drop2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=128,out_features=10)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop1(x)

       # x = self.flat(x)
        _n, _c, _h, _w = x.shape
        _x = x.permute(0, 2, 3, 1)
        # print(_x.shape)
        _x = _x.reshape(_n, _h, _w * _c)
        # print(_x.shape)
        h0 = torch.zeros(2 * 1, _n, 128).to(DEVICE)  # 初始化反馈值 num_layers * num_directions ,batch, hidden_size
        c0 = torch.zeros(2 * 1, _n, 128).to(DEVICE)
        hsn, (hn, cn) = self.rnn(_x, (h0, c0))

        #x = F.relu(self.fc1(x))
        #x = self.drop2(x)

        x = self.fc2(hsn[:, -1, :])
        output = F.log_softmax(x, dim=1)

        return output


#
# sequence_length = 28
# input_size = 13
# hidden_size = 128
# num_layers = 2
# num_classes = 10
# batch_size = 100
# num_epochs = 2
# learning_rate = 0.01


class RNN(nn.Module):
    def __init__(self):
        #, input_size, hidden_size, num_layers, num_classes, device, classes = None
        super(RNN, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(13, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 8)
        self.device = DEVICE
        #self.classes = classes
        # torch.cuda.manual_seed(42)

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # torch.cuda.manual_seed(42)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1,16)   #3,16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        # self.conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(3, 1))
        #
        # self.avg_pool = nn.AvgPool2d(2)
        self.conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(2, 1))

        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(64, num_classes)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv2d(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out