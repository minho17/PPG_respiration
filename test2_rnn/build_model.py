
import torch
import torch.nn as nn


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
    


class Correncoder_LSTM(nn.Module):
    def __init__(self,n_out,kernel_size,padding,dropout_val,batch_size,device):
        super().__init__()
        self.name = 'Correncoder_LSTM'
        self.init = 1
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

        self.num_layers = 2
        self.hidden_size = n_out[2]

        self.lstm = nn.LSTM(input_size = n_out[2], hidden_size = n_out[2],num_layers = self.num_layers, batch_first=True)

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
        # self.init_hidden(batch_size,device)

        # self.layer0 = nn.Sequential(
        #     nn.Conv1d(n_out[2], n_out[2], kernel_size=kernel_size[2], padding='same')
        # )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # out = self.layer0(out)
        out = torch.swapaxes(out, 1, 2)

        if self.init == 1:
            out, (self.h, self.c) = self.lstm(out)
            self.init = 0
        else:
            out, (self.h, self.c) = self.lstm(out,(self.h.detach(), self.c.detach()))
        
        out = torch.swapaxes(out, 1, 2)

        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
    
    # def init_hidden(self, batch_size,device):
    #     h0 = torch.zeros( (self.num_layers, batch_size, self.hidden_size), device = device) #.detach()
    #     c0 = torch.zeros( (self.num_layers, batch_size, self.hidden_size), device = device) # .detach()
    #     hidden = (h0, c0)

    #     return hidden
