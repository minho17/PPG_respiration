
import torch.nn as nn
import dsntnn2
import torch

import collections
from itertools import repeat
import torch.nn.functional as F

def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function

    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class Conv2dSame(nn.Module):
    
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)

class cnn_module1(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size,stride,is_same,is_pooling):
        super(cnn_module1,self).__init__()
        if is_same == 1:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size, stride=stride,padding='same'), 
                # nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size, stride=(1, 1)), 
                # nn.Conv1d(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size, stride=1,padding='same'), 
                # nn.InstanceNorm2d(out_dim), 
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
                # 
                # nn.InstanceNorm2d(out_dim)                                          
            )
        else:
                self.layer = nn.Sequential(
                # Conv2dSame(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size, stride=stride), 
                nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size, stride=stride), 
                # nn.Conv1d(in_channels=in_dim,out_channels=out_dim,kernel_size=kernel_size, stride=1,padding='same'), 
                nn.InstanceNorm2d(out_dim), 
                # nn.BatchNorm2d(out_dim),
                nn.ReLU()
                # 
                # nn.InstanceNorm2d(out_dim)                                          
            )
        self.is_pooling = is_pooling
        # self.maxpool = nn.MaxPool2d(kernel_size=(1,2))
        self.maxpool = nn.AvgPool2d(kernel_size=(1,2))
        

    def forward(self,x):
        out = self.layer(x)  
        
        if self.is_pooling == 1:
            out = self.maxpool(out)
        
        return out
    

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
            nn.Linear(8 * 49, 23)
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
        x2 = self.fc4(x2)
        x2 = self.fc5(x2)
        x2 = x2.unsqueeze(dim=1) 

        heatmaps = dsntnn2.flat_softmax(x2)
        coords = dsntnn2.dsnt(heatmaps)

        return out1, heatmaps, coords
    

class CNN1(nn.Module):
    def __init__(self,n_out,kernel_size,padding,dropout_val):
        super().__init__()
        self.name = 'CNN1'

        self.layer_e1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=n_out[0],kernel_size=(1,kernel_size[0]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.MaxPool2d(kernel_size=(1,2))
        ) 

        self.layer_e2 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[1],kernel_size=(1,kernel_size[1]), stride=1),
            nn.BatchNorm2d(n_out[1]),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.MaxPool2d(kernel_size=(1,2))
        ) 

        self.layer_e3 = nn.Sequential(
            Conv2dSame(in_channels=n_out[1],out_channels=n_out[2],kernel_size=(1,kernel_size[2]), stride=1),
            nn.BatchNorm2d(n_out[2]),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.MaxPool2d(kernel_size=(1,2))
        ) 


        self.layer_e4 = nn.Sequential(
            Conv2dSame(in_channels=n_out[2],out_channels=n_out[3],kernel_size=(1,kernel_size[3]), stride=1),
            nn.BatchNorm2d(n_out[3]),
            nn.ReLU(),
            nn.Dropout(dropout_val),
            nn.MaxPool2d(kernel_size=(1,2))
        ) 


        self.layer_d1 = nn.Sequential(
            nn.ConvTranspose2d(n_out[3], n_out[2], 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=n_out[2],out_channels=n_out[2],kernel_size=(1,kernel_size[3]),padding='same'), 
            nn.BatchNorm2d(n_out[2]),
            nn.ReLU()
        )

        self.layer_d2 = nn.Sequential(
            nn.ConvTranspose2d(n_out[2], n_out[1], 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=n_out[1],out_channels=n_out[1],kernel_size=(1,kernel_size[2]),padding='same'), 
            nn.BatchNorm2d(n_out[1]),
            nn.ReLU()
        )

        self.layer_d3 = nn.Sequential(
            nn.ConvTranspose2d(n_out[1], n_out[0], 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,kernel_size[1]),padding='same'), 
            nn.BatchNorm2d(n_out[1]),
            nn.ReLU()
        )

        self.layer_d4 = nn.Sequential(
            nn.ConvTranspose2d(n_out[0], 1, 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,kernel_size[0]),padding='same'), 
            nn.BatchNorm2d(1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(1,2))

    def forward(self, x):
        x = self.layer_e1(x)
        x = self.layer_e2(x)
        x = self.layer_e3(x)
        x = self.layer_e4(x)
        x = self.layer_d1(x)
        x = self.layer_d2(x)
        x = self.layer_d3(x)
        x = self.layer_d4(x)

        out = x
        return out
    


class CNN2(nn.Module):
    def __init__(self,n_out,kernel_size,kernel_size2,dropout_ratio):
        super().__init__()
        self.name = 'CNN2'

        self.layer_e1_1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=n_out[0],kernel_size=(1,kernel_size[0]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e1_2 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[0]/2)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e1_3 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[0]/4)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e1_4 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=1,kernel_size=(1,int(kernel_size[0]/8)), stride=1),
            nn.ReLU(),
        ) 

        self.layer_e2_1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=n_out[0],kernel_size=(1,kernel_size[1]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e2_2 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[1]/2)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e2_3 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[1]/4)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e2_4 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=1,kernel_size=(1,int(kernel_size[1]/8)), stride=1),
            nn.ReLU()
        )

        self.layer_e3_1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=n_out[0],kernel_size=(1,kernel_size[2]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e3_2 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[2]/2)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e3_3 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[2]/4)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e3_4 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=1,kernel_size=(1,int(kernel_size[2]/8)), stride=1),
            nn.ReLU()
        )

        self.layer_e4_1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=n_out[0],kernel_size=(1,kernel_size[3]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e4_2 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[3]/2)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e4_3 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,int(kernel_size[3]/4)), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e4_4 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=1,kernel_size=(1,int(kernel_size[3]/8)), stride=1),
            nn.ReLU()
        )

        self.layer_a1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=1,kernel_size=(1,40), stride=1),
            Conv2dSame(in_channels=1,out_channels=1,kernel_size=(1,20), stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,1))
        )

        self.layer_d1 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(1,kernel_size2[0]),padding='same'), 
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.layer_d2 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(1,kernel_size2[1]),padding='same'), 
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.layer_d3 = nn.Sequential(
            nn.ConvTranspose2d(4, 1, 1, 2, output_padding=(0,1)),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,kernel_size2[2]),padding='same'), 
            nn.BatchNorm2d(1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(1,2))

    def forward(self, x):
        x1 = self.layer_e1_1(x)
        x1 = self.layer_e1_2(x1)
        x1 = self.layer_e1_3(x1)
        x1 = self.layer_e1_4(x1)

        x2 = self.layer_e2_1(x)
        x2 = self.layer_e2_2(x2)
        x2 = self.layer_e2_3(x2)
        x2 = self.layer_e2_4(x2)

        x3 = self.layer_e3_1(x)
        x3 = self.layer_e3_2(x3)
        x3 = self.layer_e3_3(x3)
        x3 = self.layer_e3_4(x3)

        x4 = self.layer_e4_1(x)
        x4 = self.layer_e4_2(x4)
        x4 = self.layer_e4_3(x4)
        x4 = self.layer_e4_4(x4)

        x = torch.cat((x1,x2,x3,x4), dim=1)

        x = torch.swapaxes(x, 1, 2)
        a = self.layer_a1(x)

        x = x*a
        x = torch.swapaxes(x, 1, 2)

        x = self.layer_d1(x)
        x = self.layer_d2(x)
        x = self.layer_d3(x)

        out = x
        return out