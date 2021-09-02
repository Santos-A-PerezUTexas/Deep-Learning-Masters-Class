from .models import ClassificationLoss, model_factory, save_model, LinearClassifier, MLPClassifier
from .utils import accuracy, load_data
import torch
"""
********models.py has these classes:*******************************

-----ADDED THIS COMMENT 9/2/2021-------------

-----ALSO ADDED THIS COMMENT 9/2/2021, BUT LATER-------------

Added this directly on Github will it appear locally.


ClassificationLoss(torch.nn.Module)
  forward(self, input, target)

LinearClassifier(torch.nn.Module), 
  __init__, 
   forward(self, x)

MLPClassifier(torch.nn.Module), 
  __init__, 
  forward(self, x)

models.py defines these functions:

save_model(model)
load_model(model)

***************************utils.py has these classes/functions*******************

class SuperTuxDataset(Dataset)
    def __init__(self, dataset_path)
    def __len__(self)
    def __getitem__(self, idx)
    
def load_data(dataset_path, num_workers=0, batch_size=128)
def accuracy(outputs, labels)

ANd utils.py defines this (dictionary), which is imported above:

model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}



"""
def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    #raise NotImplementedError('train')
     
    x= torch.tensor([[1., -1.], [1., -1.]])
    model.forward(x)
    
    save_model(model)

     

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
