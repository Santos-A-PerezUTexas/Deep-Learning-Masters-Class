import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        CNNClassifier. Similar to homework 1, model should return a (B,6) Tensor which represents the logits of the classes. Use convolutions this time.

https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n

https://www.reddit.com/r/deeplearning/comments/hfjiog/what_is_1d2d_and_3d_convolutions_in_cnn/

A 1D convolution is for time series and it is a WxC filter that slides in the W direction. W represents the time dimension and C represents multivariate dimensions.

A 2D convolution is for images and it is a HxWxC filter that slides in the H and W directions. C indicates the Channel (RGB) for the first layer, and number of input feature maps for higher layers.

A 3D convolution is for voxel data and it is a DxHxWxC filter th
torch.nn.Conv2d
        
        Your code here
        """
        raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
