import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

        Hint: Don't be too fancy, this is a one-liner
"""         
             

class LinearClassifier(torch.nn.Module):
  
class MLPClassifier(torch.nn.Module):

  

def save_model(model):
  from torch import save
  from os import path
  for n, m in model_factory.items():
    if isinstance(model, m):
      return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    


def load_model(model):
  from torch import load
  from os import path
  r = model_factory[model]()
  r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
  return r
