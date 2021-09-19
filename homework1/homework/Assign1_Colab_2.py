#Line 78 self.Image_tensor = torch.tensor(self.Image_To_Tensor(self.one_image), requires_grad=True)         
#validation at end of training loop... accuracy()....
#Line 398 y_hat_tensor[i] = Chosen_Model(Tux_DataLoader[0][i]) #feed entire batch Sept 18
#change len() to get size of image label list??
#line 327 for image_tuples_tensor, label in Tux_DataLoader:  #Sept 18
#In Colab remove line 102 - the break statement in Dataset constructor
     

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter          #USE THIS!!!!!!!!!!!!!!!!!!   SEPT 17, 2021
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Image_Transformer
import csv     

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
input_dim = 3*64*64
n_epochs = 10                   #CHANGE EPOCHS to 100 !!!!!!!!
batch_size = 128
input_size = 64*64*3

class SuperTuxDataset(Dataset):   #kel76y

  def __init__(self, dataset_path):  
    
    self.image_list = []
    self.label_list = []
        
    val = input("PRESS ANY KEY to iterate over labels.csv and load *ALL* DATA")
    print(val) 
    
    image_index = 0
  
    with open('labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile) 
      
      for row in ImageReader:
              
        
        print(f'Image index is {image_index}, about to evaluate is bigger than 0')    
        
        if image_index > 0:
                  
          #image_file_name = "../data/train/"+row[0]  for colab Sept 18
          image_file_name = "..\data\\train\\"+row[0] 
          
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
          #self.Image_tensor = torch.tensor(self.Image_To_Tensor(self.one_image), requires_grad=True)
          
          self.image_list.append(self.Image_tensor)
          
          label_string_to_number = 0
          
          current_label_string = row[1]
           
          for i in LABEL_NAMES:  #from string to string, iterate through the strongs
            if i==current_label_string:
              self.label_list.append(label_string_to_number)
              
            label_string_to_number += 1
          
        image_index += 1 
        
        #REMOVE THIS FOR COLAB
        if image_index == 513:   #REMOVE FOR COLAB
          break
   
    print(f'Size of data set is ==============of the label list {len(self.label_list)}, and of the image list, {len(self.image_list)}')
    
    
    val = input("PRESS ANY KEY TO LEARN")
    print(val) 
    
               
  def __len__(self):
    return (len(self.label_list))  
         
  def __getitem__(self, idx):     
  
    return (self.image_list[idx], self.label_list[idx])
 
def load_data(dataset_path, num_workers=0, batch_size=batch_size):     #Use this in train.py
    
    dataset = SuperTuxDataset(dataset_path)   
    
    #length of the loader will adapt to the batch_size. So if your train dataset has 1000 samples 
    #and you use a batch_size of 10, the loader will have the length 100.
        
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)  
    
def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

"""
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
"""
   
#----------------------------------------------CLASSIFICATION LOSS      
#kel76y

class ClassificationLoss(torch.nn.Module):
  
    
  def forward(self, Y_hat_Vector, y_vector):   #OLD: def forward(self, input, target):
      
    cross_loss = torch.nn.CrossEntropyLoss()  #applies softmax to Y_hat, then crossEloss, returns mean
     
    weighted_mean_batch_loss = cross_loss(Y_hat_Vector,  y_vector)
    
    
    ##y_hat_Vector has grad_fn=<CopySlices>
    ##y_vector has NO GRAD!!!!!!!!!!!!!  PROBLEM????????

        
    return (weighted_mean_batch_loss)  #return the mean loss accross the entire batch
      
    
      
      
      
#kel76y
# -----------------------------------------------LINEAR-----------------------------------------------

class LinearClassifier(torch.nn.Module):

  def __init__(self, input_dim):        
      
    super().__init__()   #original
       
    self.input_dim = input_dim   
    self.network = torch.nn.Linear(input_dim, 6)
    
     
  def forward(self, image_tensor):   
    
    flatened_Image_tensor = image_tensor.view(image_tensor.size(0), -1).view(-1)
    #this will have NO GRADIENT!
    
    return self.network(flatened_Image_tensor).requires_grad_()   #Added requires_grad on Sept 17
   

#--------------------------------------------------------------MLP------------------------------------------------


class MLPClassifier(torch.nn.Module):  


  #learning rate, hidden_layers

  def __init__(self, hidden_size=5, input_dim=input_dim):     #set hiddensize here!!!!!!!!!!!!!!!!!!
   
    super().__init__()        
    #self.hidden_size = hidden_size

    self.network = torch.nn.Sequential( 
                torch.nn.Linear(input_dim, hidden_size),   
                torch.nn.ReLU(inplace=False),               
                torch.nn.Linear(hidden_size, 6)  
                )

                
  def forward(self, image_tensor):   
   
    print("Inside forward method of MLP")
    flattened_Image_tensor = image_tensor.view(image_tensor.size(0), -1).view(-1)   
    return (self.network(flattened_Image_tensor))

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


model_factory = { 'linear': LinearClassifier, 'mlp': MLPClassifier, }  #this has to stay here!!
  

#TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN T


def train(args):

    
    image_index = 1                   #test code
    
    Chosen_Model = model_factory[args.model](input_dim=input_dim)     #LINEAR CLASSIFIER BY DEFAULT IN THE COMMAND LINE 
     
    Tux_DataLoader =  load_data('c:\fakepath', num_workers=2)   #set num_workers here   
 
    y_hat_tensor = torch.ones(batch_size,6)  #this is going to change when put through network
    
         
    
    optimizer = torch.optim.SGD(Chosen_Model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
    
    calculate_loss = ClassificationLoss()   
    
   
    global_step = 0
    
    
    print (f'PARAMATERS FOR Chosen Model are {list(Chosen_Model.parameters())}')
      
    
    val = input(f'ABOUT TO BEGIN TRAINING, ABOVE YOU SEE THE PARAMETERSSSSSSSSSSSSSSSSSSSSSSS')
    print(val)
    
    print ("--------------------------------STARTING TRAINING---------------------------------")
    
    print (f'The size of the data loader is {len(Tux_DataLoader)}')
    
    
    val = input(f'BEGIN TRAINING LOOP (Only one Epoch Through All Batches)')
    
    print(val)
    
    #For epochs here
    
    #flatened_Image_tensor = torch.rand(input_dim)
    
#------------------------------BEGIN ITERATE THROUGH BATCHES------------------------------------------

    for batch_data, batch_labels in Tux_DataLoader:
      
      for i in range(len(batch_data)):          
        y_hat_tensor[i] = Chosen_Model(batch_data[i]) #should feed entire batch instead, Sept 18 
        
      #output = Chosen_Model(batch_data)
        
        
        
       # if (i==4):
        #  print (f'The fifth batch_data image is {batch_data[i]}, and its prediction yhat is {y_hat_tensor[i]}, this is y_hat_tensor: {y_hat_tensor}')
          
      val = input(f'!!!!!!!!!!!!!!!!About to call model_loss with this SIZE for y_hat {y_hat_tensor.size()}, and this one for target,   {batch_labels.size()}')
      print(val)
     
      print("model_loss = calculate next")
      
      model_loss = calculate_loss(y_hat_tensor, batch_labels)   
      
      print (f'Model loss is: {model_loss}')
      val = input(f'PRESS ANY KEY')
      print(val)
     
      
      print("Detect Anomaly Next")
      
      torch.autograd.set_detect_anomaly(True)
      
      print("Optimizer zero grad next")
      
      optimizer.zero_grad()
      
      print("Calling backward now to get gradients")
      #goes into LOOP here
      
      model_loss.backward(retain_graph=True)
      
      #cd cs342\homework1\homework
      
      print("Updating weights now with step()")
     
      optimizer.step()             
           
      
      model_accuracy = accuracy(y_hat_tensor, batch_labels)  #Sept 18
     
      New_model_accuracy = accuracy(y_hat_tensor, batch_labels)  #Sept 18
      print (f'Accuracy Before Weight Updates for Batch ??? is {model_accuracy}, the new one: {New_model_accuracy}')
      
      val = input(f'##################  Above you can Y_hat Tensor for both linear and MLP #################')
      print(val)
               

#------------------------------END ITERATE THROUGH BATCHES------------------------------------------


#*********************************************************END TRAINING*************************************************************
     
    model_accuracy = accuracy(y_hat_tensor, batch_labels)   #Sept 18
    print (f'Final Model Accuracy is {model_accuracy}')
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')     
    #calls the linear model by default
    # Put custom arguments here

    args = parser.parse_args()   
    
    print (f'SEPT 18--------------args is {args}')
    
     
    train(args)
    
     #Now implement MLPClassifier class. The inputs and outputs to same as the linear classifier. 
     #Try to move most modifications to command-line arguments  in ArgumentParser.
