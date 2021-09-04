from .models import LossFunction, ClassificationLoss, model_factory, save_model, LinearClassifier, MLPClassifier
from .utils import accuracy, load_data, SuperTuxDataset
import torch

"""

Original: from .utils import accuracy, load_data

train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 
train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 
train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py train.py 

Train your linear model in train.py. You should implement the full training procedure:

      1. create a model, loss, optimizer
      2. load the data: train and valid
      3. Run SGD for several epochs
      4. Save your final model, using save_model()



11111111********models.py has these classes:*******************************
11111111********models.py has these classes:*******************************

      ClassificationLoss(torch.nn.Module)
        forward(self, input, target)

      LinearClassifier(torch.nn.Module), 
        __init__, 
         forward(self, x)

      MLPClassifier(torch.nn.Module), 
        __init__, 
        forward(self, x)d

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

     https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
     https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
     Training a NN happens in two steps:

Forward Propagation: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.

Backward Propagation: Inn backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using *gradient descent*.
     
        import torch, torchvision
        model = torchvision.models.resnet18(pretrained=True)
        data = torch.rand(1, 3, 64, 64)
        labels = torch.rand(1, 1000)
        
        prediction = model(data)        # forward pass
        loss = (prediction - labels).sum()
        loss.backward()
        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        
        #Finally, we call .step() to initiate gradient descent. 
        #The optimizer adjusts each parameter by its gradient stored in .grad.

        optim.step() #gradient descent

      Loop:
        Forward Pass
        Loss
        Backward Pass - get gradients
        Update Weights

    Your code here

    """

    image_index = 1
    image = torch.rand([3,64,64]) 
    tuple1=(image, image_index)
    
    My_DataSet = SuperTuxDataset('c:\fakepath')   
    
    linear_Classifier_model = model_factory[args.model](2)  
    
      #defaults to linear
      #OR   linear_Classifier_model = LinearClassifier(2)
      #dimension of weights is 2 just example...
    
    #raise NotImplementedError('train')
     
    x = torch.rand([10,2]) 
    true_y = ((x**2).sum(1) < 1)
    
    for iteration in range(100): 
    
      Y_hat = linear_Classifier_model.forward(x)
      
      #model_loss = ClassificationLoss.forward(prediction_logit, true_y)
      
      #Y_hat_sigmoid_of_logit = 1/(1+(-prediction_logit).exp())  
      
      model_loss = LossFunction(Y_hat, true_y)
      
      print ("Model LOSS:", model_loss)
      print ("Truth Value of Logits", Y_hat > .5 )

      print (f'model loss at iteration {iteration} is {model_loss} and the prediction y_hat is {Y_hat}, while the y is {true_y}')

    
      print (f'the prediction logit converted ')
    pred_y = p_y > 0.5
    
      model_loss.backward()
    
      for p in linear_Classifier_model.parameters():                                       
    
        p.data[:] -= 0.5 * p.grad                    
        p.grad.zero_()


    print (f'*********The image:  {image}, the tuple {tuple1}')
    #save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    
    
    train(args)
    
     #Now implement MLPClassifier class. The inputs and outputs to same as the linear classifier. 
     #Try to move most modifications to command-line arguments  in ArgumentParser.
