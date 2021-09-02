import torch
import torch.nn.functional as F

"""


models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
    
class LinearClassifier(torch.nn.Module):
    def __init__(self):
    def forward(self, x):
    
class MLPClassifier(torch.nn.Module):
    def __init__(self):
    def forward(self, x):

model_factory = {'linear': LinearClassifier,    'mlp': MLPClassifier, }

def save_model(model)

def load_model(model)



"""

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

       print ("Wandavision, you're inside LinearClassifier class, __init_ constructor, models.py")
        
        """
        Your code here
        """
        #raise NotImplementedError('LinearClassifier.__init__')

        
        
    def forward(self, x):
        
        
        print ("Wandavision, you're inside LinearClassifier class, forward method, models.py")
        
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        #raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
