import torch.nn as nn

# complete model
# conv 5X5
# 32 filters
# pooling 2x2
# conv 5X5
# 64 filters
# pooling 2x2
# fc layer 10 nodes

class CNN(nn.Module):
    def __init__(self,shape):
        super(CNN,self).__init__()
        self.batch_size = shape[0]
        self.shape =shape[1:]
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.shape[0], out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # output size 
        s = (self.shape[1]-5)/1+1
        # 32,24,24
        s = int(s/2)
        # 32,12,12
        s = (s-5)/1+1
        # 64,8,8
        s = int(s/2)
        # 64,4,4

        self.fc_layer = nn.Sequential(
            nn.Linear(64*s*s,100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

    def forward(self,x):
        out = self.conv_layer(x)
        out = out.view(self.batch_size,-1)
        out = self.fc_layer(out)
        return out


class CNNLow(nn.Module):
    def __init__(self,shape):
        super(CNN,self).__init__()
        self.batch_size = shape[0]
        self.shape =shape[1:]
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.shape[0], out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # output size 
        s = (self.shape[1]-5)/1+1
        # 32,24,24
        s = int(s/2)
        # 32,12,12

        self.fc_layer = nn.Sequential(
            nn.Linear(32*s*s,100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

    def forward(self,x):
        out = self.conv_layer(x)
        out = out.view(self.batch_size,-1)
        out = self.fc_layer(out)
        return out




class CNNHigh(nn.Module):
    def __init__(self,shape):
        super(CNN,self).__init__()
        self.batch_size = shape[0]
        self.shape =shape[1:]
        

        

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # output size 
        s = (self.shape[1]-5)/1+1
        # 32,24,24
        s = int(s/2)
        # 32,12,12
        s = (s-5)/1+1
        # 64,8,8
        s = int(s/2)
        # 64,4,4

        self.fc_layer = nn.Sequential(
            nn.Linear(64*s*s,100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

    def forward(self,x):
        out = self.conv_layer(x)
        out = out.view(self.batch_size,-1)
        out = self.fc_layer(out)
        return out