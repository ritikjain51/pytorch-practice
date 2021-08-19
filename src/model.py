import torch 
import torch.nn as nn

class ClassificationModel(nn.Module):
    
    def __init__(self, out_dim, channel_first = False):
        super(ClassificationModel, self).__init__()
        self.input = nn.Conv2d(3, 256, (3, 3), padding="valid")
        self.conv1 = nn.Conv2d(256, 128, (3, 3))
        self.maxp1 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, out_dim)
        
    def forward(self, x):
        x = self.input(x) # in: n x 3 x 150 x 150   out: n x 1024 x 148 x 148
        x = self.conv1(x) # in: n x 1024 x 148 x 148  out: n x 512 x 146 x 146
        x = F.relu(self.maxp1(x)) # in: n x 512 x 146 x 146  out: n x 512 x 73 x 73
        x = x.view((-1, 128*73*73))
        x = F.relu(self.fc1(x)) # in: n x 2728448 out: 1024
        return F.softmax(self.out(x))
