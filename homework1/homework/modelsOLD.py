import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html

"""

OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION 
OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION 
OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION OLD VERSION 


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

def LossFunction(Y_hat_Vector, y_vector):
  
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

"""

def LossFunction (prediction_logit, y_vector):
  
  Y_hat_Vector = 1/(1+(-prediction_logit).exp()) 
  
  return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

class ClassificationLoss(torch.nn.Module):

    
    def forward(Y_hat_Vector, y_vector):   #OLD: def forward(self, Y_hat_Vector, y_vector):
        
      return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      
      #This is the negative log likelihood for logistic regression, need softmax instead.
      #In Logistic Regression, Y_hat_vector is a prediction for ALL x(i) in the data set, so it returns
      #a vector of "i" scalars.  In Softmax, this would return a vector (tensor) of "i" vectors - not scalars).
        
        
"""
        
        Implement the log-likelihood of a softmax classifier.

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')



        Hint: Don't be too fancy, this is a one-liner
"""
         
             
        #raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim):        #input_dim parameter not needed for homework!
      
    super().__init__()
    self.w = Parameter(torch.zeros(input_dim))
    self.b = Parameter(-torch.zeros(1))
    print ("Wandavision, you're inside LinearClassifier class, __init_ constructor, models.py")

  def forward(self, x):      
        
    print ("Wandavision, you're inside LinearClassifier class, forward method, models.py")      
    return (x * self.w[None,:]).sum(dim=1) + self.b 


"""
        Your code here
"""
#raise NotImplementedError('LinearClassifier.__init__')

        
        
        
"""
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
"""
        #raise NotImplementedError('LinearClassifier.__init__')        
        #raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):

  def __init__(self):
    super().__init__()
    
  def forward(self, x):
      
      #  @x: torch.Tensor((B,3,64,64))
      #  @return: torch.Tensor((B,6))
      

"""
        MLPClassifier class. The inputs and outputs to same as the linear classifier. 
        However, now you’re learning a non-linear function.

        Train your network using python3 -m homework.train -m mlp
          
    Hint: Some tuning of your training code. Try to move most modifications to command-line arguments 
    in ArgumentParser

    Hint: Use ReLU layers as non-linearities.

    Hint: Two layers are sufficient.

    Hint: Keep the first layer small to save parameters.

    You can test your trained model using python3 -m grader homework -v
        
        SUMMARY:
        
      
        MLPClassifier class. The inputs and outputs to same as the linear classifier. 
        Some tuning of your training code. Use ArgumentParser
        Use ReLU layers as non-linearities.
        Two layers are sufficient.
        Keep the first layer small to save parameters.
        
"""    

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
