import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, num_inputs=784, num_hidden=1024, num_classes=10):
        super().__init__()                
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)        
        self.fc3 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax(1)
        
    def forward(self, x):                
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
