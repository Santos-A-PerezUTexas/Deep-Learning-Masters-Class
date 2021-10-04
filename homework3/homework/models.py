
  #HOMEWORK 3

import torch
import torch.nn.functional as F

"""
1.   TUNE THE CNNClassifier 


  -Input normalization - https://en.wikipedia.org/wiki/Batch_normalization?
  -Residual blocks (see video on residual connections pytorch programming by professor)
  -Dropout
  -Data augmentations (Both geometric and color augmentations are important. Be aggressive here. Different levels of supertux have radically different lighting.)
  -Weight regularization
  -Early stopping


Input Normatlization

  http://www.philkr.net/dl_class/lectures/making_it_work/09.pdf

Augmentation:

  https://drive.google.com/file/d/1XOYF7qxlj3sDPk0dzV5w2jo4THvspQkk/view
  https://discuss.pytorch.org/t/data-augmentation-in-pytorch/7925
  https://www.youtube.com/watch?v=Zvd276j9sZ8



Residual Connections:

  https://drive.google.com/file/d/1in6SpWW0pRCE_aibvG2jCUvphlCt0wdp/view

  http://www.philkr.net/dl_class/lectures/making_it_work/18.html
  (this link also has batch normalization)

  http://www.philkr.net/dl_class/lectures/making_it_work/17.pdf

Identity Mapping for Residual Connections:

  https://openreview.net/pdf?id=ryxB0Rtxx
  reparameterization of the convolutional layers such that when all trainable weights
  are 0, the layer represents the identity function. Formally, for an input x, each 
  residual layer has the form x + h(x), rather than h(x)

"""

class CNNClassifier(torch.nn.Module):
   def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case
       
      self.layer1 = torch.nn.Sequential(
      
            torch.nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2),
            #torch.nn.BatchNorm2d(32),
            #torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                      )    
            
      self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))                  
        
              
      self.fc1 = torch.nn.Linear(32 * 32 * 32, 6)   
      
       #self.drop_out = nn.Dropout()
          
    
    def forward(self, images_batch):
    
     
      out = self.layer1(images_batch)
      out = out.reshape(out.size(0), -1)
      #out = self.drop_out(out)
      out = self.fc1(out)
      
                   
      return out


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

                """                                        

        https://github.com/pochih/FCN-pytorch/tree/master/python
        https://github.com/wkentaro/pytorch-fcn/tree/master/torchfcn
        https://medium.com/@iceberg12/semantic-segmentation-applied-on-car-road-4ee62622292f
        
        https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
        https://discuss.pytorch.org/t/add-residual-connection/20148/6
        https://stats.stackexchange.com/questions/321054/what-are-residual-connections-in-rnns
        
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        
        """
        raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r