#Loss can be > 1
#validation at end of training loop... accuracy()....
#Line 398 y_hat_tensorLinear[i] = linear_M(Tux_DataLoader[0][i]) #feed entire batch Sept 18
#change len() to get size of image label list??
#line 327 for image_tuples_tensor, label in Tux_DataLoader:  #Sept 18
     

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
  
  
    #self.BatchSize = 5120  colab   #SHOULD NOT USE HERE!
    self.BatchSize = 512
      
    #NO BATCH SIZE...  Use a list...  
    
    
    #Maybe remove self???
    self.imageDATASET = torch.rand([self.BatchSize,3,64,64])   #COMPLETE BUT RANDOM DATA SET
    
    self.image_list = []
    self.label_list = []
    
    #list has no set size.... can append... Sept 18
    #argument parser  Sept 18
    
     
    val = input("PRESS ANY KEY TO BEGIN")
    print(val) 
    
    
    print ("I will now iterate over labels.cvs file.........")
    
    val = input("PRESS ANY KEY to iterate over labels.csv and load *ALL* DATA")
    print(val) 
    
    
    image_index = 0
  
    with open('labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile) 
      
      for labelsFILE_image_row in ImageReader:
              
        if image_index == 0:
          print (f'image index is zero')
          image_file_name = "..\data\\train\\"+labelsFILE_image_row[0]
          print (f'This is the image name at zero:  {image_file_name}')
          val = input("Press any key")
          print(val)
          
        if image_index > 0:
                  
          #image_file_name = "../data/train/"+labelsFILE_image_row[0]  for colab Sept 18
          #print(image_file_name)  commented Sept 17 evening
          image_file_name = "..\data\\train\\"+labelsFILE_image_row[0] 
          
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
  
          self.imageDATASET[image_index-1] = self.Image_tensor  #added -1 to image_index Sept 18   
          self.image_list.append(self.Image_tensor)
          
          label_string_to_number = 0
          
          #iterate through label names, assign current label[image_index] a number:
          current_label_string = labelsFILE_image_row[1]
           
          for i in LABEL_NAMES:  #from string to string, iterate through the strongs
            if i==current_label_string:
              self.label_list.append(label_string_to_number)
              #self.labels[image_index-1] = label_string_to_number
            label_string_to_number += 1
          
                  
          #label_list.append(self.labels[image_index-1])   #WARNING -1 ????
          
          
          #print (f'I just assigned self.labels[{image_index-1}] the value {self.labels[image_index-1]} which corresponds to label {labelsFILE_image_row[1]}')
          
        image_index += 1 
        if image_index == self.BatchSize+1:   #added +1 Sept 18
          break
   
    print(f'ABOVE IS THE LIST.  Size of data set is =============={self.imageDATASET.size(0)}, of the label list {len(self.label_list)}, and of the image list, {len(self.image_list)}')
    
    
    val = input("PRESS ANY KEY TO LEARN")
    print(val) 
    
    print (self.imageDATASET[0])
    
    val = input("Above is the first image tensor from the data set, does it have grad?")
    print(val) 
    
    
    
    

 
            
  def __len__(self):
    return (self.imageDATASET.size(0))  
         
  def __getitem__(self, idx):     
  
    #return (self.imageDATASET[idx], self.labels[idx]) #image Tensor size (3,64,64) range [0,1], label is int.
    return (self.image_list[idx], self.label_list[idx])
        
#kel76y

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

  def __init__(self, hidden_size):   
   
    super().__init__()        
    self.hidden_size = hidden_size
    
    #added this Sept 17, 2021
    #self.w = Parameter(torch.ones(input_dim))
    #self.b = Parameter(-torch.ones(1))
        
    self.linear1 = torch.nn.Linear(input_dim, hidden_size)
    torch.nn.init.normal_(self.linear1.weight, std=0.01)
    torch.nn.init.normal_(self.linear1.bias, std=0.01)
    self.linear2 = torch.nn.Linear(hidden_size, 6)
    self.ReLU = torch.nn.ReLU(inplace=False)
        
    self.network = torch.nn.Sequential( 
                torch.nn.Linear(input_dim, hidden_size),   
                torch.nn.ReLU(inplace=False),               
                torch.nn.Linear(hidden_size, 6)  
                )
                
  def forward(self, image_tensor):   
   
    
    flattened_Image_tensor = image_tensor.view(image_tensor.size(0), -1).view(-1)   
    return self.linear2(self.ReLU(self.linear1(flattened_Image_tensor)))  #added this Sept 17, revert back????
    

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
    
    #linear_M = model_factory[args.model](2)     #LINEAR CLASSIFIER BY DEFAULT IN THE COMMAND LINE 
    linear_M = LinearClassifier(input_dim)
    MLPx = MLPClassifier(hidden_size=5)                                     
     
    Tux_DataLoader =  load_data('c:\fakepath', num_workers=2)   #set num_workers here   
 
    y_hat_tensor = torch.ones(batch_size,6)  #this is going to change when put through network
    
    #y_hat_tensorLinear = torch.ones(batch_size, 6)           Commented Sept 18
    #y_hat_tensorMLP = torch.ones(batch_size,6)              #requires_grad = True????
      
    
                    
        
 
#trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    
    
    
    #create the optimizer (change to use args??)
    #just defining the optimizer, call it w/ step to update weights 
    #but must implement loss.backward first to get the gradients
    
    optimizerMLPx = torch.optim.SGD(MLPx.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    optimizerLinear = torch.optim.SGD(linear_M.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
    
    # Create the loss - For these batch_size images, the network predicted the labels as set forth by y_hat_vector 
    #(change to tensor!!!)
    
    calculate_loss = ClassificationLoss()   
    
    #call it with Y_hat_Vector, y_vector... input (y_hat predicted labels) and 
    #target (y_vector actual image label)
    #now I can call it as such, model_loss = loss_object (y_hat_tensor, y_tensor) where y_tensor has actual values.. 
    #or is it tuples???  
    

    
    #BEGIN SGD kel76y
    
    global_step = 0
    
    
    print (f'PARAMATERS FOR MLPx are {list(MLPx.parameters())}')
    print (f'PARAMATERS FOR Linear are {list(linear_M.parameters())}')
      
    
    val = input(f'ABOUT TO BEGIN TRAINING, ABOVE YOU SEE THE PARAMETERSSSSSSSSSSSSSSSSSSSSSSS')
    print(val)
    
    print ("--------------------------------STARTING TRAINING---------------------------------")
    
    print (f'The size of the data loader is {len(Tux_DataLoader)}')
    
    
    val = input(f'BEGIN TRAINING LOOP (Only one Epoch Through All Batches)')
    
    print(val)
    
    #For epochs here
    
    #flatened_Image_tensor = torch.rand(input_dim)
    
#------------------------------BEGIN ITERATE THROUGH BATCHES------------------------------------------

    #for image_tuples_tensor, label in Tux_DataLoader:  #Sept 18
    for batch_idx, image_tuples_tensor in enumerate(Tux_DataLoader):  #iterate batches
    
    #for image_tuples_tensor, label in Tux_DataLoader:  #Sept 18
    
      print (f'image_tuples_tensor[1], the labels for this batch[{batch_idx}], is {image_tuples_tensor[1]} ')
      
      val = input(f'At beggining of for loop [{batch_idx}] for batches, iterating through all batches')
      print(val)
      
      #------------------------------BEGIN ITERATE THROUGH ALL IMAGES OF A BATCH
  
      y_hat_tensorLinear = linear_M(image_tuples_tensor[0]) #feed entire batch Sept 18       
      y_hat_tensorMLP = MLPx(image_tuples_tensor[0])   #flattening occurs in MLPx
  
      #y_hat_tensorLinear = linear_M(image_tuples_tensor[0][0:64]) #feed entire batch Sept 18       
      #y_hat_tensorMLP = MLPx(image_tuples_tensor[0][0:64])   #flattening occurs in MLPx
  

   
      #for i in range(len(image_tuples_tensor[0])):          
        #y_hat_tensorLinear[i] = linear_M(image_tuples_tensor[0][i]) #feed entire batch Sept 18       
        #y_hat_tensorMLP[i] = MLPx(image_tuples_tensor[0][i])   #flattening occurs in MLPx
        
    #-------------------------END ITERATE THROUGH ALL IMAGES OF A BATCH --------------------------
      
          
      val = input(f'!!!!!!!!!!!!!!!!About to call model_loss with this SIZE for y_hat {y_hat_tensorMLP.size()}, and this one for target,   {image_tuples_tensor[1].size()}')
      print(val)
     
      
      model_lossLinear = calculate_loss(y_hat_tensorLinear, image_tuples_tensor[1])   
      model_lossMLP = calculate_loss(y_hat_tensorMLP, image_tuples_tensor[1])#8x8x8x8x8x8x8x8x8x8x8x8x8x8x8x8  HERE
      
      torch.autograd.set_detect_anomaly(True)
      
 
      optimizerMLPx.zero_grad()
      
      optimizerLinear.zero_grad()
      
      model_lossLinear.backward(retain_graph=True)
      
      #model_lossMLP.backward(retain_graph=True)         
     
      optimizerLinear.step()             
      optimizerMLPx.step()                #UNCOMMENT THIS SEPT 17
      
      
      
      model_accuracy = accuracy(y_hat_tensorLinear, image_tuples_tensor[1])  #Sept 18
      print (f'Accuracy Before Weight Updates for Batch {batch_idx} is {model_accuracy}')
      for i in range(len(image_tuples_tensor[0])):  #iterate through all images, MAYBE I DONT NEED THIS!
        y_hat_tensorLinear[i] = linear_M(image_tuples_tensor[0][i]) 
        y_hat_tensorMLP[i] = MLPx(image_tuples_tensor[0][i])
      New_model_accuracy = accuracy(y_hat_tensorLinear, image_tuples_tensor[1])  #Sept 18
      print (f'Accuracy Before Weight Updates for Batch {batch_idx} is {model_accuracy}, the new one: {New_model_accuracy}')
      
      val = input(f'##################  Above you can Y_hat Tensor for both linear and MLP #################')
      print(val)
               

#------------------------------END ITERATE THROUGH BATCHES------------------------------------------


#*********************************************************END TRAINING*************************************************************
     
    model_accuracy = accuracy(y_hat_tensorLinear, image_tuples_tensor[1])   #Sept 18
    print (f'Final Model Accuracy is {model_accuracy}')
    

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')     #calls the linear model by default
    # Put custom arguments here

    args = parser.parse_args()   
     
    train(args)
    
     #Now implement MLPClassifier class. The inputs and outputs to same as the linear classifier. 
     #Try to move most modifications to command-line arguments  in ArgumentParser.
