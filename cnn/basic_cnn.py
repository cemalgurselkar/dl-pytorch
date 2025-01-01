import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Basic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #Conv1 --> ReLu --> Pool
        x = self.pool(F.relu(self.conv2(x))) #Conv2 --> ReLu --> Pool
        x = x.view(-1, 64*7*7) # Flatten
        x = F.relu(self.fc1(x)) # FC1 -> ReLu
        x = self.dropout(x) #Dropout
        x = self.fc2(x) #FC2
        return x

model = CNN_Basic()
input = torch.randn(16,1,28,28) #input size 28x28
output = model(input)
print(output.shape)