import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv     #utils.py

iterations_for_sgd = 10

#UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY 


LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):

  def __init__(self, dataset_path):
  
    self.imageDATASET = torch.rand([2,3,64,64]) 
    self.size = 64,64
    self.one_image = Image.open(r"sample_image.jpg")
    print ("Just opened the sample image, about to show it to you.")
    #self.one_image.show()
          
  def __len__(self):
    return (3000)  
         
  def __getitem__(self, idx):     
    return (self.imageDATASET, LABEL_NAMES[idx])
  
  def get_item(self, idx):     
    return(self.__getitem__(idx))
    
  def get_fake_image(self, idx):     
    return(self.imageDATASET[idx])

  def get_real_image(self, idx):     
    return (self.one_image)
    
  
def load_data(dataset_path, num_workers=0, batch_size=128):     #use this in train.py
    
    dataset = SuperTuxDataset(dataset_path)
    with open('labels.csv', newline='') as csvfile:
      ImageReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      #cvs.reader returns a reader object which will iterate over lines in the given csvfile
      for row in ImageReader:
        print(', '.join(row))
        print (f'ROW IS -------------------{row}')
    
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)
    
def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
  

def LossFunction (prediction_logit, y_vector):
  Y_hat_Vector = 1/(1+(-prediction_logit).exp()) 
  return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

class ClassificationLoss(torch.nn.Module):
     
  def forward(Y_hat_Vector, y_vector):   #OLD: def forward(self, Y_hat_Vector, y_vector):
        
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

class LinearClassifier(torch.nn.Module):

#Your forward function receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor (one value per class).
#B is the batch size, it's a hyper parameter we can set.

  def __init__(self, input_dim):        #input_dim parameter not needed for homework!
      
    super().__init__()
    self.w = Parameter(torch.zeros(input_dim))
    self.b = Parameter(-torch.zeros(1))
    print ("Wandavision, you're inside LinearClassifier class, __init_ constructor, models.py")

  def forward(self, x):      
        
    print ("Wandavision, you're inside LinearClassifier class, forward method, models.py")      
    return (x * self.w[None,:]).sum(dim=1) + self.b 
 

class MLPClassifier(torch.nn.Module):
  def __init__(self):
    super().__init__()
  #def forward(self, x):
      
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


model_factory = { 'linear': LinearClassifier, 'mlp': MLPClassifier, }
  
  
#train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 


def train(args):

    image_index = 1
    image = torch.rand([3,64,64]) 
    tuple1=(image, image_index)
    
    My_DataSet = SuperTuxDataset('c:\fakepath')   
    
    
    My_Real_DataSet = load_data('../data/train', num_workers=0, batch_size=128)
    
    
    image_dataSET = My_DataSet.get_item(2)
    
    linear_Classifier_model = model_factory[args.model](2)     #DEFINING THE CLASSIFIER HERE     
    x = torch.rand([10,2]) 
    true_y = ((x**2).sum(1) < 1)
    
    #iterations_for_sgd defined at the beggining
    for iteration in range(iterations_for_sgd): 
    
      Y_hat = linear_Classifier_model.forward(x)
      
      #model_loss = ClassificationLoss.forward(prediction_logit, true_y)
      
      #Y_hat_sigmoid_of_logit = 1/(1+(-prediction_logit).exp())  
      
      model_loss = LossFunction(Y_hat, true_y)
      
      print ("Model LOSS:", model_loss)
      print ("Truth Value of Logits", Y_hat > .5 )

      print (f'model loss at iteration {iteration} is {model_loss} and the prediction y_hat is {Y_hat}, while the y is {true_y}')

     
      model_loss.backward()
    
      for p in linear_Classifier_model.parameters():                                       
    
        p.data[:] -= 0.5 * p.grad                    
        p.grad.zero_()


    print (f'*********The local random image:  {image}, the local tuple {tuple1}')
    
    print (f'22222222222222222---->The  image DATASET tuple {image_dataSET}')
    
    fake_Image = My_DataSet.get_fake_image(1)
    real_Image = My_DataSet.get_real_image(1)
    

    print (f'33333333333333333333---->A fake image {fake_Image}')
    print (f'44444444444444444444---->A real image {real_Image}')
    
    print (f'Just did {iterations_for_sgd} Gradient Descent iterations with ten UNIT CIRCLE points ONLY!!!')
    
    
    #GRADIENT DESCENT USING THE  IMAGES----------------------------------------------------
    #LOAD THE 250 IMAGES 
    

    
    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    
    
    train(args)
    
     #Now implement MLPClassifier class. The inputs and outputs to same as the linear classifier. 
     #Try to move most modifications to command-line arguments  in ArgumentParser.
