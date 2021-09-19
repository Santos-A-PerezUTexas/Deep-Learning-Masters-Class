from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch



LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):

  def __init__(self, dataset_path):
  
    self.imageDATASET = torch.rand([2,3,64,64]) 
    self.size = 64,64
    self.one_image = Image.open(r"./homework/sample_image.jpg")
    self.one_image.show()
        
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
    
        
        
       

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
