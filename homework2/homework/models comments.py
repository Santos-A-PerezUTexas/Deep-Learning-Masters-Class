import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        CNNClassifier. Similar to homework 1, model should return a (B,6) Tensor which represents the logits of the classes. Use convolutions this time.

https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n

https://www.reddit.com/r/deeplearning/comments/hfjiog/what_is_1d2d_and_3d_convolutions_in_cnn/

    A 1D convolution is for time series and it is a WxC filter that slides in the W direction. W represents 
    the time dimension and C represents multivariate dimensions.

    A 2D convolution is for images and it is a HxWxC filter that slides in the H and W directions. 
    C indicates the Channel (RGB) for the first layer, and number of input feature maps for higher layers.

    A 3D convolution is for voxel data and it is a DxHxWxC filter th torch.nn.Conv2d
            
https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

        Your code here
def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6)
        )
    
                    COMMENTS
                    ---------
    I think we can use an example here, assume we have a small image [3(channels),8(height),8(width)]?[3,8,8]
We want to do 5 classifications for each pixel in this image. So it will be: lnput[3×8×8]?Output[8×8×5],

If we use linear classification, it will be a linear layer: linear[input:3,output:5].

If we use 1×1conv, it will be  conv2d(in_channel:3,out_channel:5

http://www.philkr.net/dl_class/lectures/convolutional_networks/03.html
https://drive.google.com/file/d/1I9Hp8q-_Z8oDYCKqvX-A8ZRh5kdFHWds/view
https://drive.google.com/file/d/1I9Hp8q-_Z8oDYCKqvX-A8ZRh5kdFHWds/view

https://www.youtube.com/watch?v=p1xZ2yWU1eo

https://datascience.stackexchange.com/questions/64278/what-is-a-channel-in-a-cnn

A convolution layer receives the image (w×h×c) as input, and generates as output an activation map of dimensions w'× h'× c'.  (NOTICE HOW c disappears - because "The f×f×c kernel is element-wise multiplied by the input matrix window, and the resulting elements are added together into a single value.")  And we have c' instead because "normally you want to apply not just one single f×f×c filter, but many different filters. The number of filters you apply is c'." 


        """

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        
        def forward(self, xb):
        return self.network(xb)        

        """
        

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
