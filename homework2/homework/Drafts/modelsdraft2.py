import torch


class CNNClassifier(torch.nn.Module):

    def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
        super().__init__()
        
        
        c = n_input_channels
        
        self.network = torch.nn.Sequential( 
        
                torch.nn.Conv2d(c, 1, kernel_size=1),   
                torch.nn.ReLU(inplace=False),               
                torch.nn.(hidden_size, 6)  
        
                )
        
        
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size))
            L.append(torch.nn.ReLU())
            c = l
        L.append(torch.nn.Conv2d(c, 1, kernel_size=1))
        
        self.layers = torch.nn.Sequential(*L)
    
    def forward(self, x):
        return self.layers(x).mean(dim=[1,2,3])
        

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
