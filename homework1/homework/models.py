import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


#----------------------------------------------CLASSIFICATION LOSS      


class ClassificationLoss(torch.nn.Module):
  
    
  def forward(self, Y_hat_Vector, y_vector):   #OLD: def forward(self, input, target):
      
    cross_loss = torch.nn.CrossEntropyLoss()  #applies softmax to Y_hat, then crossEloss, returns mean
     
    weighted_mean_batch_loss = cross_loss(Y_hat_Vector,  y_vector)
        
    return (weighted_mean_batch_loss)  #return the mean loss accross the entire batch
      
    
      

# -----------------------------------------------LINEAR-----------------------------------------------

class LinearClassifier(torch.nn.Module):

  def __init__(self, input_dim=3*64*64):        
      
    self.input_dim = input_dim
    super().__init__()   
       
    self.network = torch.nn.Linear(input_dim, 6)            
    
    
     
  def forward(self, image_tensor):   
   
    batch_size=128
    input_dim = self.input_dim

    
    temp_tensor = torch.zeros(batch_size, 6)
    #reshaped = image_tensor.reshape(batch_size, input_dim)

    for i in range(batch_size):
      #temp_tensor[i] = self.network(image_tensor[i].view(image_tensor[i].size(0), -1).view(-1))
      #temp_tensor[i] = self.network(reshaped[i]) 
      temp_tensor[i] = self.network(image_tensor[i].reshape(1, input_dim))
    return (temp_tensor) 

    #return (self.network(image_tensor.view(image_tensor.size(0), -1).view(-1))) 
    
#--------------------------------------------------------------MLP------------------------------------------------

class MLPClassifier(torch.nn.Module):  

  def __init__(self, input_dim=12288, hidden_size=5):     #set hiddensize here
   
    super().__init__() 
    
    self.input_dim = input_dim      

    self.network = torch.nn.Sequential( 
                torch.nn.Linear(input_dim, hidden_size),   
                torch.nn.ReLU(inplace=False),               
                torch.nn.Linear(hidden_size, 6)  
                )

                
  def forward(self, image_tensor):   

    batch_size=128
    input_dim = self.input_dim

    temp_tensor = torch.zeros(batch_size, 6)
    reshaped = image_tensor.reshape(batch_size, input_dim)

    for i in range(batch_size):
      #temp_tensor[i] = self.network(image_tensor[i].view(image_tensor[i].size(0), -1).view(-1))
      temp_tensor[i] = self.network(reshaped[i]) 
    return (temp_tensor) 
          
    #return (self.network(image_tensor))

model_factory = { 'linear': LinearClassifier, 'mlp': MLPClassifier, } 

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


 #this has to stay here!!

