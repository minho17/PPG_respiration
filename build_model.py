

import torch.nn as nn
import dsntnn

class Correncoder_model(nn.Module):
    def __init__(self,n_out,kernel_size,padding,dropout_val):
        super().__init__()
        self.name = 'Correncoder'
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, n_out[0], kernel_size=kernel_size[0], padding=padding[0]),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out[0], n_out[1], kernel_size=kernel_size[1], padding=padding[1]),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out[1], n_out[2], kernel_size=kernel_size[2], padding=padding[2]),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out[2], n_out[1], kernel_size=kernel_size[2], padding=padding[2]),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out[1], n_out[0], kernel_size=kernel_size[1], padding=padding[1]),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out[0], 1, kernel_size=kernel_size[0], padding=padding[0])
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
    
class Correncoder_model2(nn.Module):
    def __init__(self,n_out,kernel_size,padding,dropout_val):
        super().__init__()
        self.name = 'Correncoder2'
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, n_out[0], kernel_size=kernel_size[0], padding=padding[0]),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out[0], n_out[1], kernel_size=kernel_size[1], padding=padding[1]),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out[1], n_out[2], kernel_size=kernel_size[2], padding=padding[2]),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out[2], n_out[1], kernel_size=kernel_size[2], padding=padding[2]),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out[1], n_out[0], kernel_size=kernel_size[1], padding=padding[1]),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out[0], 1, kernel_size=kernel_size[0], padding=padding[0])
        )
        
        self.fc1 = nn.Sequential(
            nn.Conv1d(n_out[2], n_out[2], kernel_size=kernel_size[2], padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Conv1d(n_out[2], n_out[2], kernel_size=kernel_size[2], padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc3 = nn.Sequential(
            nn.Conv1d(n_out[2], n_out[2], kernel_size=kernel_size[2], padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc4 = nn.Sequential(
            nn.Conv1d(n_out[2], n_out[2], kernel_size=kernel_size[2], padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.fc5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 49, 23),
            heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps),
            coords = dsntnn.dsnt(heatmaps)
        )


    def forward(self, x):
        x0 = self.layer1(x)
        x0 = self.layer2(x0)
        x0 = self.layer3(x0)
        
        x1 = self.layer4(x0)
        x1 = self.layer5(x1)
        out1 = self.layer6(x1)
        
        x2 = self.fc1(x0)
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)
        out2 = self.fc4(x2)
        return out1, out2