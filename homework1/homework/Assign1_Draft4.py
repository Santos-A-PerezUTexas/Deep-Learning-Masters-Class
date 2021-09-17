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

  
  

class SuperTuxDataset(Dataset):   #kel76y

  def __init__(self, dataset_path):
  
  
    self.BatchSize = 512
    self.imageDATASET = torch.rand([self.BatchSize,3,64,64])   #COMPLETE BUT RANDOM DATA SET
 
     #self.labels = torch.ones(self.BatchSize)
     
    self.labels = torch.randint(0, 5, (self.BatchSize, ))
    
    
    val = input("PRESS ANY KEY AND BE BOLD")
    print(val) 
    
    
    print ("I will now iterate over labels.cvs file.........")
    
    val = input("PRESS ANY KEY to iteraTE over labels.csv and load ALL DATA")
    print(val) 
    
    
    image_index = 0
  
    with open('labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile) 
      #ImageReader = csv.reader(csvfile, delimiter=' ', quotechar='|') 
      #https://www.geeksforgeeks.org/working-csv-files-python/     
      
      for labelsFILE_image_row in ImageReader:
        
        #print(', '.join(row))
        #print (f'File names ONLY-------------------{row[0]}')
        #print (f'Label names only=================={row[1]}')
        
        if image_index > 0:
                  
          image_file_name = "..\data\\train\\"+labelsFILE_image_row[0] 
          print(image_file_name)
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
          self.imageDATASET[image_index] = self.Image_tensor
          
          label_string_to_number = 0
          
          #iterate through label names, assign current label[image_index] a number:
          current_label_string = labelsFILE_image_row[1]
           
          for i in LABEL_NAMES:  #from string to string, iterate through the strongs
            if i==current_label_string:
              self.labels[image_index-1] = label_string_to_number
            label_string_to_number += 1
             
          print (f'I just assigned self.labels[{image_index-1}] the value {self.labels[image_index-1]} which corresponds to label {labelsFILE_image_row[1]}')
          
        image_index += 1 
        if image_index == self.BatchSize:
            print(f'I will now break, image_index is {image_index}')
            break
    

    print(f'The size of this data set is ==================={self.imageDATASET.size(0)}')
    
    val = input("There!  I just LOADED ALL DATA up to Batch size!")
    print(val) 
    
         
        
  def __len__(self):
    return (self.imageDATASET.size(0))  
         
  def __getitem__(self, idx):     
    return (self.imageDATASET[idx], self.labels[idx]) #image Tensor size (3,64,64) range [0,1], label is int.
    
        
#kel76y

def load_data(dataset_path, num_workers=0, batch_size=batch_size):     #Use this in train.py
    #all this does is return the data_loader to use in SGD?
  
    #https://machinelearningknowledge.ai/pytorch-dataloader-tutorial-with-example/
    
    dataset = SuperTuxDataset(dataset_path)   #In Orginal
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
      
    Get_Softmax = torch.nn.Softmax()  #may have to change this to logsoftmax, then use nllloss 
    Negative_L_loss = torch.nn.NLLLoss()
    cross_loss = torch.nn.CrossEntropyLoss()
 
    #nn.CrossEntropyLoss() is actually a combination of nn.LogSoftmax() and nn.NLLLoss(). Refer this 5 doc.
    #weighted_mean_batch_loss = Negative_L_loss(Get_Softmax(Y_hat_Vector),  y_vector)
    #weighted_mean_batch_loss = Negative_L_loss(Y_hat_Vector,  y_vector)
 
    weighted_mean_batch_loss = cross_loss(Y_hat_Vector,  y_vector)
    
    
    #Y_hat_vector is a minibatch with 128 6-tensor entries for 128 (batch_size) image outputs
    #y_vector is the a 128 1-tensor vector with the correct class label
    #Get_log_Softmax returns 128 6-tensor entries with 6 log probabilities for each image
    #Negative_L_Loss returns a 1 tensor entry with the mean? loss????, with backward you get gradients, then optimize.step
        
    print (f'*********INSIDE Classification Loss, Y_hat_Vector vector is {Y_hat_Vector}, and this compares with the actual values{y_vector}')
    
    print (f'*********STILL INSIDE Classification Loss, the weighted_mean_batch_loss is {weighted_mean_batch_loss}')
    
    #print (f'*********The backward fx of weighted_mean_batch_loss (gradient) is {weighted_mean_batch_loss.backward(retain_graph=True)}')
        
        
    return (weighted_mean_batch_loss)  #return the mean loss accross the entire batch
      
    
      
      
      
#kel76y
# -----------------------------------------------LINEAR-----------------------------------------------

class LinearClassifier(torch.nn.Module):

  def __init__(self):        
      
    super().__init__()   #original
       
    self.input_dim = input_dim
    
    self.network = torch.nn.Linear(input_dim, 6)
    
     
  def forward(self, image_tensor):   
    
    flatened_Image_tensor = image_tensor.view(image_tensor.size(0), -1).view(-1)
    
    return self.network(flatened_Image_tensor)
  


#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP*************
#kel76y

class MLPClassifier(torch.nn.Module):  

#The inputs and outputs to the multi-layer perceptron are the same as the linear classifier.
#Some tuning of your training code. Move modifications to command-line arguments in ArgumentParser


  def __init__(self, hidden_size):   
   
    super().__init__()        
    self.hidden_size = hidden_size
        
    self.network = torch.nn.Sequential( 
                torch.nn.Linear(input_dim, hidden_size),   #----->keep this layer small to save parameters???
                torch.nn.ReLU(inplace=False),               
                torch.nn.Linear(hidden_size, 6)  
                )
                
  def forward(self, image_tensor):   
  
    flatened_Image_tensor = image_tensor.view(image_tensor.size(0), -1).view(-1)
    
    return self.network(flatened_Image_tensor)
  
  #def forward(self, multiple_image_tensor):   
  #receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor
       
  

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
    My_DataSet = SuperTuxDataset('c:\fakepath')      
    #linear_M = model_factory[args.model](2)     #LINEAR CLASSIFIER BY DEFAULT IN THE COMMAND LINE 
    linear_M = LinearClassifier()
    MLPx = MLPClassifier(hidden_size=5)                                     
     
    Tux_DataLoader =  load_data('c:\fakepath') 
 
    y_hat_tensor = torch.ones(batch_size,6)  #this is going to change when put through network
    
    y_hat_tensorLinear = torch.ones(batch_size, 6)
    y_hat_tensorMLP = torch.ones(batch_size,6)              #requires_grad = True????
      
    real_Image_tensor, real_image_label = My_DataSet[0]   #gets image [0] TENSOR  tuple
                    
        
 
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
    
   
    print ("--------------------------------STARTING TRAINING---------------------------------")
    
    print (f'The size of the data loader is {len(Tux_DataLoader)}')
    
    #For epochs here
    
    #flatened_Image_tensor = torch.rand(input_dim)
    
    for batch_idx, image_tuples_tensor in enumerate(Tux_DataLoader):  #iterate batches
           
      for i in range(len(image_tuples_tensor[0])):  #iterate through all images, MAYBE I DONT NEED THIS!
        #print (f'label {i}, for batch {batch_idx} is {image_tuples_tensor[1][i]}')
        
        y_hat_tensorLinear[i] = linear_M(image_tuples_tensor[0][i]) 
        
        y_hat_tensorMLP[i] = MLPx(image_tuples_tensor[0][i])
        
        #y_hat_tensorMLP[i] = MLPx(flatened_Image_tensor)
        #y_hat_tensorMLP[i] = MLPx(flatened_Image_tensor)   "a view requires grad in place operation"
        #y_hat_tensorMLP[i] = MLPx(flatened_Image_tensor)
        
        print (f'%%%%%%%%  Just assinged y_hat_tensor[{i}] this value:   {MLPx(image_tuples_tensor[0][i])}')
      
      #end iterate through all images
      
      #print ('batch idx{}, batch len {}'.format(batch_idx, len(image_tuples_tensor)))
      #print (f'------------This is batch {batch_idx}, it has size {len(image_tuples_tensor)}, and here is the image_tuples_tensor: {image_tuples_tensor}')
      #print (f'------------The size of image_tuples_tensor[0] for batch[{batch_idx}], the images, is {len(image_tuples_tensor[0])}')
      #print (f'------------The size of image_tuples_tensor[1] for batch[{batch_idx}], the labels, is {len(image_tuples_tensor[1])}')
      #print (f'These are the labels for this batch:  {image_tuples_tensor[1]}')
           
      val = input(f'!!!!!!!!!!!!!!!!About to call model_loss with this value for y_hat {y_hat_tensorMLP}, and this one for target,   {image_tuples_tensor[1]}')
      print(val)
      
      
      model_loss = calculate_loss(y_hat_tensorMLP, image_tuples_tensor[1]) 
      
      torch.autograd.set_detect_anomaly(True)
      
      optimizerMLPx.zero_grad()
      model_loss.backward(retain_graph=True) 
      optimizerMLPx.step()
        
      print (f'==============================LOSS FOR BATCH {batch_idx} is {model_loss} and the gradient is deleted')
            
      print ("#### NOW calculate the loss, obtain gradients, and update weights BEFORE next batch")
      
      val = input(f'##################  Above you can see batch {batch_idx}  #################')
      print(val)
      
      print(f'y_hat_tensorLinear is {y_hat_tensorLinear}')
      
      print(f'y_hat_tensorMLP is {y_hat_tensorMLP}')
      
      
      
      #MLP_gradient = y_hat_tensorMLP.backward()
      #linear_gradient = y_hat_tensorLinear()
      
      #need to calculate the loss first!
      #print (f'The gradient for linear is {linear_gradient}, and for MLP its {MLP_gradient}')
      #need to calculate the loss first! and take backward of the loss!
      #print ("BOTH OF THESE ARE WRONG, NEED TO TRAIN NOW")
      
    
      val = input(f'##################  Above you can Y_hat Tensor for both linear and MLP #################')
      print(val)
               
        
    #for epoch in range(n_epochs): 
    
      # Shuffle the data
      #permutation = torch.randperm(Local_Fake_Image_Set.size(0))  #generate batch_size numbers from 0 to batch_size-1
     
      
      #model_loss = LossFunction(Y_hat, true_y)                    
      #model_loss.backward()   #get the gradients with the computation graph
    
      
      #for p in linear_Classifier_model.parameters():         #update parameters, replace with optimizer step                
    
        #p.data[:] -= 0.5 * p.grad                    
        #p.grad.zero_()

#*********************************************************END TRAINING*************************************************************
#*********************************************************END TRAINING*************************************************************
#*********************************************************END TRAINING*************************************************************
#*********************************************************END TRAINING*************************************************************

      

    print ("--------------------------------FINISHED TRAINING---------------------------------")
    
  
    #print (f'Here is the permutation iterative which goes in the for loop, len(permutation)-batch_size+1 = {len(permutation)} minus {batch_size+1}')
    
     
    #SEPT 12, 2021:  __get_item__ is called when you create a new image object from the dataset object!
     
    
    #save_model(model)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')     #calls the linear model by default
    # Put custom arguments here

    args = parser.parse_args()   
     
    train(args)
    
     #Now implement MLPClassifier class. The inputs and outputs to same as the linear classifier. 
     #Try to move most modifications to command-line arguments  in ArgumentParser.
