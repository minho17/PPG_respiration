import torch.nn as nn
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


class model(nn.Module):
    def __init__(self,n_out,kernel_size,kernel_size2,dropout_ratio):
        super().__init__()
        self.name = 'model'

        self.layer_e1 = nn.Sequential(
            Conv2dSame(in_channels=1,out_channels=n_out[0],kernel_size=(1,kernel_size[0]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e2 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,kernel_size[1]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
        ) 

        self.layer_e3 = nn.Sequential(
            Conv2dSame(in_channels=n_out[0],out_channels=n_out[0],kernel_size=(1,kernel_size[2]), stride=1),
            nn.BatchNorm2d(n_out[0]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout_ratio)
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

    def forward(self, x):

        x1 = self.layer_e1(x)
        x1 = self.layer_e2(x1)
        x = self.layer_e3(x1)

        x = self.layer_d1(x)
        x = self.layer_d2(x)
        x = self.layer_d3(x)

        out = x
        return out
