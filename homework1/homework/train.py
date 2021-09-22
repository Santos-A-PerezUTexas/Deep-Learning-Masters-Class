import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter          
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Image_Transformer
import csv     

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
input_dim = 3*64*64
n_epochs = 10                   #CHANGE EPOCHS to 100 !!!!!!!!
batch_size = 128
input_size = 64*64*3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Train_data_path = "..\data\\train\\"
Train_data_path = "./data/train/"
#For colab path is "./data/train/"

#Test_data_path = "..\data\\valid\\"
Test_data_path = "../data/valid/"



class SuperTuxDataset(Dataset):   

  def __init__(self, dataset_path):  
    
    self.image_list = []
    self.label_list = []
        
    image_index = 0
  
    with open(dataset_path+'labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile) 
      
      for row in ImageReader:
                     
        if image_index > 0:
      
          image_file_name = dataset_path+row[0]
          
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
                    
          self.image_list.append(self.Image_tensor)
          
          label_string_to_number = 0
          
          current_label_string = row[1]
           
          for i in LABEL_NAMES:  #from string to string, iterate through the strongs
            if i==current_label_string:
              self.label_list.append(label_string_to_number)
              
            label_string_to_number += 1
          
        image_index += 1 
        
        #REMOVE THIS FOR COLAB
        #if image_index == 513:   #REMOVE FOR COLAB
         # break
               
  def __len__(self):
    return (len(self.label_list))  
         
  def __getitem__(self, idx):     
  
    return (self.image_list[idx], self.label_list[idx])
 
def load_data(dataset_path, num_workers=0):     #Use this in train.py
    
    dataset = SuperTuxDataset(dataset_path)      
       
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


class ClassificationLoss(torch.nn.Module):
  
    
  def forward(self, Y_hat_Vector, y_vector):   #OLD: def forward(self, input, target):
      
    cross_loss = torch.nn.CrossEntropyLoss()  #applies softmax to Y_hat, then crossEloss, returns mean
     
    weighted_mean_batch_loss = cross_loss(Y_hat_Vector,  y_vector)
        
    return (weighted_mean_batch_loss)  #return the mean loss accross the entire batch
      
    
      

# -----------------------------------------------LINEAR-----------------------------------------------

class LinearClassifier(torch.nn.Module):

  def __init__(self, input_dim):        
      
    super().__init__()   #original
       
    self.network = torch.nn.Linear(input_dim, 6)
    
     
  def forward(self, image_tensor):   
   
     return (self.network(image_tensor))   

#--------------------------------------------------------------MLP------------------------------------------------

class MLPClassifier(torch.nn.Module):  

  def __init__(self, input_dim, hidden_size=5):     #set hiddensize here
   
    super().__init__()        

    self.network = torch.nn.Sequential( 
                torch.nn.Linear(input_dim, hidden_size),   
                torch.nn.ReLU(inplace=False),               
                torch.nn.Linear(hidden_size, 6)  
                )

                
  def forward(self, image_tensor):   
       
    return (self.network(image_tensor))

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

    
    image_index = 1                   
    
    Chosen_Model = model_factory[args.model](input_dim=input_dim)     
     
    Tux_DataLoader =  load_data(Train_data_path, num_workers=2)   
    
    optimizer = torch.optim.SGD(Chosen_Model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
    
    calculate_loss = ClassificationLoss()   
    
   
    global_step = 0
    
     
    print ("--------------------------------STARTING TRAINING---------------------------------")
    
    print (f'The size of the data loader is {len(Tux_DataLoader)}')
    
    
    for epochs in range(n_epochs): # set to n_epochs in colab
    
#------------------------------BEGIN ITERATE THROUGH BATCHES------------------------------------------

      for batch_data, batch_labels in Tux_DataLoader:


        reshaped = batch_data.reshape(len(batch_data), input_dim)
        #reshaped has no gradient  Sept 19
              
        predictions = Chosen_Model(reshaped)   #substituted reshaped for batch_data
        #predictions has gradient  grad_fn=<AddmmBackward> Sept 19
        #batch_labels has no gradient Sept 19
       
        model_loss = calculate_loss(predictions, batch_labels)
      
        torch.autograd.set_detect_anomaly(True)
      
        optimizer.zero_grad()
      
        model_loss.backward(retain_graph=True)
      
        print("Updating weights now with step()")
     
        optimizer.step()             
           
        model_accuracy = accuracy(predictions, batch_labels)  #Sept 18
     
        print (f'Model Accuracy is  {model_accuracy}')
      
   
#------------------------------END ITERATE THROUGH BATCHES------------------------------------------


#*********************************************************END TRAINING*************************************************************
     
    model_accuracy = accuracy(predictions, batch_labels)   #Sept 18
    print (f'Final Model Accuracy is {model_accuracy}')
    save_model(Chosen_Model)
    
    
    #Test_DataLoader =  load_data(Test_data_path, num_workers=2, batch_size=batch_size) 
    #TestData, TestLabels =  Test_DataLoader()
    #TestPredictions = Chosen_Model(TestData)
    #model_accuracy = accuracy(TestPredictions, TestLabels)   #Sept 18
    #print (f'Model Accuracy on TEST DATA is {model_accuracy}')
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')     
    #calls the linear model by default
    # Put custom arguments here

    args = parser.parse_args()   
    
    print (f'SEPT 18--------------args is {args}')
    print ("IN NEW TRAIN")
     
    train(args)
  
