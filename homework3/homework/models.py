
#HOMEWORK 3
#Oct 13, 2021
#no pooling
#Always pad by kernel_size / 2, use an odd kernel_size
#Oct 13:  DO I HAVE TO use transforms on the labels?
#Oct 13 - does randcrop 64 do anything?
#OCT 13 NIGHT:  TOOK OUT BLOCK OF FCN
#Oct 13 Night:  Added target = target.type(torch.LongTensor) to Loss
#REMEMBER TO SET PADDING TO DIVIDED BY 2!!!!!!!!!!!!!!!!
#Apply augmentations like ColorJitter() and RandomHorizontalFlip() in train.py.
#Apply augmentations like ColorJitter() and RandomHorizontalFlip() in train.py.
#Apply augmentations like ColorJitter() and RandomHorizontalFlip() in train.py.
#Apply augmentations like ColorJitter() and RandomHorizontalFlip() in train.py.
 
#don’t want a reduction in resolution you can set your
#padding to (KS-1)/2 (same as //2 in python) and then stride by 1. 
#CHANGE PADDING TO 5/2??????????????????????
                         

import torch
import torch.nn.functional as F


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
     
  class BlockUP(torch.nn.Module):
      
    def __init__(self, n_input, n_output, stride=1):
      super().__init__()
            
      self.net = torch.nn.Sequential(
              torch.nn.ConvTranspose2d(n_input, n_output, kernel_size= (5,5), stride=(1,1), padding=(2,2), dilation=1, output_padding=(0,0)),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
            )
            
      self.downsample = torch.nn.Sequential(
              
              torch.nn.Conv2d(n_input, n_output, 1),
              torch.nn.BatchNorm2d(n_output),
              )
        
    def forward(self, x):
            
        identity = self.downsample(x)
        return self.net(x) + identity
            
  class BlockDown(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
      super().__init__()
            
      self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=5, padding=2, stride=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
            )
            
      self.downsample = torch.nn.Sequential(
              
              torch.nn.Conv2d(n_input, n_output, 1),
              torch.nn.BatchNorm2d(n_output),
              )
        
    def forward(self, x):
            
            #need identity from PRIOR level?
            identity = self.downsample(x)  #original image size, different channels
            return self.net(x) + identity  #Trying to add two different sizes, shit.
       
      
      #input is  ([32, 3, 96, 128])
      
  def __init__(self, layers=[], n_input_channels=3, kernel_size=3):  #FCN Constructor

      super().__init__()

      self.layer1 = torch.nn.Conv2d(3, 32, kernel_size=(5,5), stride=(1,1), padding=(2, 2), dilation=1, groups=1)
         
      #Output is  ([32, 32, 96, 128])  <------IDENTITY!!!!!
      self.encoder1 = self.BlockDown(32, 64) 
      #x is now ([32, 64, 48, 64])

      self.encoder2 = self.BlockDown(64, 128)
     #Output  is  ([32, 128, 24, 32]) 

      self.encoder3 = self.BlockDown(128, 256)
      #Output  is  ([32, 256, 12, 16])

      self.encoder4 = self.BlockDown(256, 512)
      #Output  is  ([32, 512, 6, 8])

      self.to_classes = torch.nn.Conv2d(5, 5, kernel_size=1) 
     
      #Output  is  ([32, 5, 6, 8])


      self.decoder = self.BlockUP(128, 5) 

      self.decoderA = self.BlockUP(5, 5) 
      ## Output is ([32, 5, 12, 16])

      self.decoderB = self.BlockUP(5, 5) 
      ## Output is ([32, 5, 24, 32])

      self.decoderC = self.BlockUP(5, 5)
      ## Output is ([32, 5, 48, 64])

      self.decoderD = self.BlockUP(5, 5)
      ## Output is ([32, 5, 96, 128])


      #input should be ([32, 128, 24, 32]) HERE.  OCT 16, 2021

               

  def forward(self, x):
             
      #Input x is [32, 3, 96, 128]
      out = self.layer1(x) #out.shape is ([32, 32, 96, 128])  <------IDENTITY
     
      #out is now [32, 32, 96, 128]
      out = self.encoder1(out)  
      
      #out is now ([32, 64, 48, 64]), 64 channels & downsampled
      out = self.encoder2(out)  
      #out  is  ([32, 128, 24, 32]), 128 channels & downsampled 
      #out = self.encoder3(out)  
      #out  is ([32, 256, 12, 16]) 
      #out = self.encoder4(out)  
      #out  is  ([32, 512, 6, 8]),

      out=self.decoder(out)  #128 to five channels

      out = self.to_classes(out)  #5 to 5 channels kernel 1
      
      #out  is  ([32, 5, 6, 8]), 

      #out=self.decoderA(out)
      ## Output is ([32, 5, 12, 16])

      #out=self.decoderB(out)
      ## Output is ([32, 5, 24, 32])

      #out=self.decoderC(out)
      ## Output is ([32, 5, 48, 64])
      
      #out=self.decoderD(out)  
      ## Output is ([32, 5, 96, 128])

    
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
