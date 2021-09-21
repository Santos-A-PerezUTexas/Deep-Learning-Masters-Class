import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
      super().__init__()

      self.network = torch.nn.Sequential(
                  
                  torch.nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
                  torch.nn.ReLU(),
                  torch.nn.MaxPool2d(2,2),
              
                  torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
                  torch.nn.ReLU(),
                  torch.nn.MaxPool2d(2,2),
                  
                  torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
                  torch.nn.ReLU(),
                  torch.nn.MaxPool2d(2,2),
                  
                  torch.nn.Flatten(),
                  torch.nn.Linear(82944,1024),
                  torch.nn.ReLU(),
                  torch.nn.Linear(1024, 512),
                  torch.nn.ReLU(),
                  torch.nn.Linear(512,6)
              )
    
    def forward(self, x):
     return self.network(x)
  
        

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
