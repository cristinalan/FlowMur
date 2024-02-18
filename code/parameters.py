import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
LOG_INTERVAL = 20
BATCH_SIZE = 256
EPOCH = 1000


Target_Label = 0

Learning_Rate_Benign_Model = 0.0001

Patience_Benign_Model = 20

Trigger_Initial_Value = 0.1

Epsilon = 0.2

Learning_Rate_Generate_Trigger = 0.0001

Patience_Generate_Trigger = 20

Poison_Rate = 0.05

Snr_DB = 30

Learning_Rate_Poisoned_Model = 0.0001

Patience_Poisoned_Model = 20