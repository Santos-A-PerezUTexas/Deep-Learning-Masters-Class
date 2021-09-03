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

    linear_Classifier_model = model_factory[args.model](2)  
      #defaults to linear
      #OR   linear_Classifier_model = LinearClassifier(2)
      #dimension of weights is 2 just example...
    
    #raise NotImplementedError('train')
     
    x = torch.rand([25,2]) 
    true_y = ((x**2).sum(1) < 1)
    
    for iteration in range(100): 
    
      prediction_logit = linear_Classifier_model.forward(x)
      model_loss = ClassificationLoss(prediction_logit, true_y)
      model_loss.backward()
    
      for p in linear_Classifier_model.parameters():                                       
    
        p.data[:] -= 0.5 * p.grad                    
        p.grad.zero_()
    
    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    
    
    train(args)
