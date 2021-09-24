import torch

#https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
#calculate dimensions https://docs.google.com/spreadsheets/d/1UUromHy5ksKlcFafpJ760HoZ-vEwAJcoCKG0U80K5L8/edit#gid=0

"""
Formula [(W-K+2P)/S]+1.

W is the input volume - in your case 128  (image size, my case 64)
K is the Kernel size - in your case 5
P is the padding - in your case 0 i believe
S is the stride - which you have not provided.

"""
class CNNClassifier(torch.nn.Module):

    def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case
       
      self.layer1 = torch.nn.Sequential(
      
            torch.nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2),#32 output channels, 3 input channels, output is 64x64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))    #This produces 32x32 output 
            
      self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  #32 input channels from layer1 (32x32 dims also), 64 output channels
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))                  #down-sampling, or pooling, to produce a 32 x 32 output of layer 2 (reducing from 64x64).
        
      #self.drop_out = nn.Dropout()
        
      self.fc1 = torch.nn.Linear(32 * 32 * 64, 100)   #this takes 32x32 of layer 2
        
      self.fc2 = torch.nn.Linear(100, 6)      #10 OUTPUTS, Changed to 6
    
    
    def forward(self, images_batch):
    
     
      out = self.layer1(images_batch)
      out = self.layer2(out)
      out = out.reshape(out.size(0), -1)
      #out = self.drop_out(out)
      out = self.fc1(out)
      out = self.fc2(out)
                   
      return out
    #Expected 4-dimensional input for 4-dimensional weight [32, 3, 5, 5], but got 3-dimensional input of size [3, 64, 64] instead 
    #The error is happening because you are feeding the images to the convolution one by one. The convolution is expecting a tensor of the size
     #(batch_size, channels, height, width). Processing the images in a loop will also be inefficient since the computation does not run in parallel this way.

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
