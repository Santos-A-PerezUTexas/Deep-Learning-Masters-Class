import torch

#https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/

class CNNClassifier(torch.nn.Module):

    def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case
       
      self.layer1 = nn.Sequential(
      
            nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2),    #32 output channels, c input channels, output is 14x14 for 28x28 image!
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
      self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  #32 input channels from layer1, 64 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))                  #down-sampling, or pooling, to produce a 7 x 7 output of layer 2.
        
      #self.drop_out = nn.Dropout()
        
      self.fc1 = nn.Linear(7 * 7 * 64, 1000)   #this takes 7x7 of layer 2
        
      self.fc2 = nn.Linear(1000, 6)      #10 OUTPUTS, Changed to 6
    
    def forward(self, x):
    
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.reshape(out.size(0), -1)
      #out = self.drop_out(out)
      out = self.fc1(out)
      out = self.fc2(out)
    
      return out
        

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
