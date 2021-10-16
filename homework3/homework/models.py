
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
 
            
  #return(torch.cat(output, identity))
            

  def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
      c = n_input_channels    #3 in our case
      
      k=2
      p=0
      s=2

      #input is  ([32, 3, 96, 128])
      self.layer1 =torch.nn.Conv2d(3, 32, kernel_size=(5,5), stride=(1,1), padding=(2, 2), dilation=1, groups=1)
      self.bn32 = torch.nn.BatchNorm2d(32)
      self.Relu = torch.nn.ReLU() 
      #Output is  ([32, 32, 96, 128])  <------IDENTITY!!!!!
      self.encoder1 = torch.nn.Conv2d(32, 64, kernel_size=(k,k), padding=(p,p), stride=(s,s), bias=False)
      self.bn64 = torch.nn.BatchNorm2d(64)
      #Relu      
      #x is now ([32, 64, 48, 64])

      self.encoder2 = torch.nn.Conv2d(64, 128, kernel_size=(2,2), padding=(0,0), stride=(2,2), bias=False)
      self.bn128 = torch.nn.BatchNorm2d(128)
      #Relu         
      #Problem is above (2,2) kernel  <-------------------*
      #Crash "Calculated padded input size per channel: (1 x 8). 
      #Kernel size: (2 x 2). Kernel size can't be greater than actual input size" ]
      #Output  is  ([32, 128, 24, 32]) 

      self.encoder3 = torch.nn.Conv2d(128, 256, kernel_size=(2,2), padding=(0,0), stride=(2,2), bias=False)
      self.bn256 = torch.nn.BatchNorm2d(256)
      #Relu         
      #Problem is above (2,2) kernel  <-------------------*
      #Crash "Calculated padded input size per channel: (1 x 8). 
      #Kernel size: (2 x 2). Kernel size can't be greater than actual input size" ]
      #Output  is  ([32, 256, 12, 16])

      self.encoder4 = torch.nn.Conv2d(256, 512, kernel_size=(2,2), padding=(0,0), stride=(2,2), bias=False)
      self.bn512 = torch.nn.BatchNorm2d(512)
      #Relu         
      #Problem is above (2,2) kernel  <-------------------*
      #Crash "Calculated padded input size per channel: (1 x 8). 
      #Kernel size: (2 x 2). Kernel size can't be greater than actual input size" ]
      #Output  is  ([32, 512, 6, 8])

      self.encoder5 = torch.nn.Conv2d(512, 5, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=False)
      self.bn5 = torch.nn.BatchNorm2d(5)
      #Relu         
      #Problem is above (2,2) kernel  <-------------------*
      #Crash "Calculated padded input size per channel: (1 x 8). 
      #Kernel size: (2 x 2). Kernel size can't be greater than actual input size" ]
      #Output  is  ([32, 5, 6, 8])


      self.decoderA = torch.nn.ConvTranspose2d(5, 5, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1))
      #batchnorm5
      #Relu
      ## Output is ([32, 5, 12, 16])

      self.decoderB = torch.nn.ConvTranspose2d(5, 5, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1))
      #batchnorm5
      #Relu
      ## Output is ([32, 5, 24, 32])

      self.decoderC = torch.nn.ConvTranspose2d(5, 5, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1))
      #batchnorm5
      #Relu
      ## Output is ([32, 5, 48, 64])

      self.decoderD = torch.nn.ConvTranspose2d(69, 5, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1))
      #batchnorm5
      #Relu
      ## Output is ([32, 5, 96, 128])



      #input should be ([32, 128, 24, 32]) HERE.  OCT 16, 2021

      self.decoder1 = torch.nn.ConvTranspose2d(128, 64, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1))
      #batchnorm64
      #Relu
      ## Output is ([32, 64, 48, 64])
      
      self.decoder2 = torch.nn.ConvTranspose2d(64, 32, kernel_size= (3,3), stride=(2,2), padding=(1,1), dilation=1, output_padding=(1,1))
      #Output is ([32, 32, 96, 128])  <--add identity to this.
               
      self.final_conv = torch.nn.Conv2d(8, 5, kernel_size=1) #([32, 5, 96, 128])  #note: channels 8, not 32 or 35


  def forward(self, x):
             
      #Input x is [32, 3, 96, 128]
      out = self.layer1(x) #out.shape is ([32, 32, 96, 128])  <------IDENTITY
      identity1 = out  #identity shape is 
      out = self.bn32(out)
      out = self.Relu(out)
     
      #out is now [32, 32, 96, 128]
      out = self.encoder1(out)  
      identity2 = out
      out = self.bn64(out)
      out = self.Relu(out)  
      #out is now ([32, 64, 48, 64]), 64 channels & downsampled
      identity1 = out

      out = self.encoder2(out)  
      out = self.bn128(out)  
      out = self.Relu(out)
      #out  is  ([32, 128, 24, 32]), 128 channels & downsampled 


      out = self.encoder3(out)  
      out = self.bn256(out)  
      out = self.Relu(out)
      #out  is ([32, 256, 12, 16]) 


      out = self.encoder4(out)  
      out = self.bn512(out)  
      out = self.Relu(out)
      #out  is  ([32, 512, 6, 8]), 


      out = self.encoder5(out)  
      out = self.bn5(out)  
      out = self.Relu(out)
      #out  is  ([32, 5, 6, 8]), 

      out=self.decoderA(out)
      out = self.bn5(out)
      out = self.Relu(out)
      ## Output is ([32, 5, 12, 16])

      out=self.decoderB(out)
      out = self.bn5(out)
      out = self.Relu(out)
      ## Output is ([32, 5, 24, 32])

      out=self.decoderC(out)
      out = self.bn5(out)
      out = self.Relu(out)
      ## Output is ([32, 5, 48, 64])
      skip_connection2 = torch.cat((out, identity1), 1) #identity1 is [32, 64, 48, 64], 69 channels total

      out=self.decoderD(skip_connection2)  #69 channels in!
      out = self.bn5(out)
      out = self.Relu(out)
      ## Output is ([32, 5, 96, 128])

      skip_connection3 = torch.cat((x, out), 1)  #[3 channels in x, plus 5 channels in out = 8 channels totals]  

      out = self.final_conv(skip_connection3) #8 channels in



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
