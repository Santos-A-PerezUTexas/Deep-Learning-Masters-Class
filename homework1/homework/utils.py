from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Image_Transformer
import csv     
import torch

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
batch_size = 128

class SuperTuxDataset(Dataset):   

  def __init__(self, dataset_path):  
    
    self.image_list = []
    self.label_list = []
        
    val = input("PRESS ANY KEY to iterate over labels.csv and load *ALL* DATA")
    print(val) 
    
    image_index = 0
  
    with open('labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile) 
      
      for row in ImageReader:
                     
        if image_index > 0:
      
          image_file_name = dataset_path+row[0]
          
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
          #self.Image_tensor = torch.tensor(self.Image_To_Tensor(self.one_image), requires_grad=True) sets gradient but crashes
          
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
               
  def __len__(self):
    return (len(self.label_list))  
         
  def __getitem__(self, idx):     
  
    return (self.image_list[idx], self.label_list[idx])
 
def load_data(dataset_path, num_workers=0, batch_size=batch_size):     #Use this in train.py
    
    dataset = SuperTuxDataset(dataset_path)      
       
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)  
    
def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()