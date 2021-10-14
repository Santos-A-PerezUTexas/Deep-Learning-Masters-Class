
  #HOMEWORK 3
  #Oct 13, 2021
  #no pooling
  #Always pad by kernel_size / 2, use an odd kernel_size
  #Oct 13:  DO I HAVE TO use transforms on the labels?
  #Oct 13 - does randcrop 64 do anything?
  #OCT 13 NIGHT:  TOOK OUT BLOCK OF FCN
  #Oct 13 Night:  Added target = target.type(torch.LongTensor) to Loss
  #REMEMBER TO SET PADDING TO DIVIDED BY 2!!!!!!!!!!!!!!!!


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

"""
class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input, target)
        
#########################CNN  BEGIN

class CNNClassifier(torch.nn.Module):

  class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity


  def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case

     
      self.layer1 = torch.nn.Sequential(
      
            torch.nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            self.Block(32,32),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                      )    
            
      self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))                  
        
              
      self.fc1 = torch.nn.Linear(32 * 32 * 32, 6)   
      
      self.drop_out = torch.nn.Dropout()
          
    
  def forward(self, images_batch):
   
     
      out = self.layer1(images_batch)
      out = out.reshape(out.size(0), -1)
      out = self.drop_out(out)
      out = self.fc1(out)
      
                   
      return out

######################################### END CNN

#########################################BEGIN FCN

class FCN(torch.nn.Module): 
  class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.ConvTranspose2d(n_input, n_output, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1)),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
           
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            output = self.net(x)
            #print (f'The size of output is {output.shape}, the size of identity is {identity.shape}')
            if self.downsample is not None:
                identity = self.downsample(x)
            return output #+ identity
            #return(torch.cat(output, identity))
            #RETURN TORCH.CAT???

  def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case

     
      self.layer1 = torch.nn.Sequential(
      
            #CHANGE PADDING TO 5/2??????????????????????
            
            torch.nn.Conv2d(3, 32, kernel_size=(5,5), stride=(1,1), padding=(2, 2), dilation=1, groups=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=2, padding=0, stride=2, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=2, padding=0, stride=2, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),   #image is now 16x16.....NO!, orginal image is 128x96!!
            #don’t want a reduction in resolution you can set your padding to (KS-1)/2 (same as //2 in python) and then stride by 1. 
                       
                                     )    
        
      self.layer2 = torch.nn.Sequential( 
        
        torch.nn.ConvTranspose2d(128, 64, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1)),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        self.Block(64,32),   #input is 32x32...  Out is 64x64, CAN add identity     
        torch.nn.Conv2d(32, 5, kernel_size=1)
        
                      )
  def forward(self, images_batch):
      
      print(f'In FCN, the size of input x is {images_batch.shape}')
      out = self.layer1(images_batch)
      print(f'After layer 1, encoder, the images of x is {out.shape}')
      out = self.layer2(out)
      print(f'After layer 2,decoder, the images of x is {out.shape}')
             
      return out

###################END FCN###############################


        #Simply, our goal is to take either a RGB color image (height×width×3) or a grayscale
        # image (height×width×1) and output a segmentation map where each pixel contains a
        #class label represented as an integer (height×width×1).

        #Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        
        #Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
        #     if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
        #     convolution
       
        

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
