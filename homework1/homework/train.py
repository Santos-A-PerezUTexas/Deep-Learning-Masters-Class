import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter          
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Image_Transformer
import csv   
from .utils import accuracy, load_data  
from .models import ClassificationLoss, model_factory, save_model

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
    
    
    for epochs in range(n_epochs-8): # set to n_epochs in colab
    
#------------------------------BEGIN ITERATE THROUGH BATCHES------------------------------------------
      idx = len(Tux_DataLoader)

      for batch_data, batch_labels in Tux_DataLoader:

        print(f'->Epoch {epochs}, batch {idx}')
      
        idx -= 1
        if idx==1:
          break
        predictions = Chosen_Model(batch_data)      
        #predictions = Chosen_Model(reshaped)   #substituted reshaped for batch_data
        #predictions has gradient  grad_fn=<AddmmBackward> Sept 19
        #batch_labels has no gradient Sept 19
       
        model_loss = calculate_loss(predictions, batch_labels)
      
        torch.autograd.set_detect_anomaly(True)
      
        optimizer.zero_grad()
      
        model_loss.backward(retain_graph=True)
      
        #print("Updating weights now with step()")
     
        optimizer.step()             
           
        #model_accuracy = accuracy(predictions, batch_labels)  #Sept 18
        
        #print (f'Model Accuracy is  {model_accuracy}')
      
   
#------------------------------END ITERATE THROUGH BATCHES------------------------------------------


#*********************************************************END TRAINING*************************************************************
     
    train_data, train_labels = next(iter(Tux_DataLoader))

    #train_data, train_labels = batch_data, batch_labels
    
    print (f'data size {train_data.size()}, labels size {train_labels.size()}')

    predictions = Chosen_Model(train_data)
    model_accuracy = accuracy(predictions, train_labels)   #Sept 18
    #print (f'Final Model Accuracy is {model_accuracy}')
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
  
