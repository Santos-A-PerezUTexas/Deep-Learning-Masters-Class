from .models import ClassificationLoss, model_factory, save_model, LinearClassifier, MLPClassifier
from .utils import accuracy, load_data
import torch
"""

train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 
train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 
train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 



11111111********models.py has these classes:*******************************
11111111********models.py has these classes:*******************************

      ClassificationLoss(torch.nn.Module)
        forward(self, input, target)

      LinearClassifier(torch.nn.Module), 
        __init__, 
         forward(self, x)

      MLPClassifier(torch.nn.Module), 
        __init__, 
        forward(self, x)

      save_model(model)
      load_model(model)

      model_factory = {'linear': LinearClassifier, 'mlp': MLPClassifier }


22222222***************************utils.py has these classes/functions*******************
22222222***************************utils.py has these classes/functions*******************

      class SuperTuxDataset(Dataset)
          def __init__(self, dataset_path)
          def __len__(self)
          def __getitem__(self, idx)
          
      def load_data(dataset_path, num_workers=0, batch_size=128)
      def accuracy(outputs, labels)




"""
def train(args):

    """

      Loop:
        Forward Pass
        Loss
        Backward Pass - get gradients
        Update Weights


    Your code here

    """

    model = model_factory[args.model]()
    
    #raise NotImplementedError('train')
     
    x = torch.rand([25,2]) 
    True_y = ((x**2).sum(1) < 1)
    
    
    model.forward(x)
    #defaults to linear
    
    
    model_loss = ClassificationLoss(p_y)
    
    model_loss.backward()
    
    
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
