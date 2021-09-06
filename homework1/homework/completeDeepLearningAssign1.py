import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


"""
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY 
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path)
    def __len__(self):
    def __getitem__(self, idx): 



def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
    

"""

"""

https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

This change made DIRECTLY on Github.  THIS change 9/1/2021

As a first step, we will need to implement a data loader for the SuperTuxKart dataset. Complete the __init__, __len__, and the __getitem__ of the SuperTuxDataset class in the utils.py.
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pillow.readthedocs.io/en/stable/reference/Image.html
https://pillow.readthedocs.io/en/stable/reference/Image.html
https://www.geeksforgeeks.org/python-pil-image-open-method/


The __len__ function should return the size of the dataset.

The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.

Labels and the corresponding image paths are saved in labels.csv, their headers are *file* and *label*. 
There are 6 classes of objects. Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.


Hint: We recommendd using the csv package to read csv files and the PIL library (Pillow fork) to read images in Python.

Hint: Use torchvision.transforms.ToTensor() to convert the PIL image to a pytorch tensor.

Hint: You have (at least) two options on how to load the dataset. 
You can load all images in the __init__ function, or you can lazily load them in __getitem__.
If you load all images in __init__, make sure you convert the image to a tensor in the constructor, otherwise, you might get an OSError: [Errno 24] Too many open files.


"""


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
    
        
  """
  
  You can load all images in the __init__ function, or you can lazily load them in __getitem__.
        If you load all images in __init__, make sure you convert the image to a tensor in the constructor      
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
       *****Hint: Use the python csv library to parse labels.csv
        https://docs.python.org/3/library/csv.html#module-csv

        WARNING: Do not perform data normalization here. 
        
        STDataset = pd.read_csv('data/labels.csv')

        
        
       ********* SAMPLE CODE*************************************
        landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

         n = 65
        img_name = landmarks_frame.iloc[n, 0]
        landmarks = landmarks_frame.iloc[n, 1:]
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.astype('float').reshape(-1, 2)

        print('Image name: {}'.format(img_name))
        print('Landmarks shape: {}'.format(landmarks.shape))
        print('First 4 Landmarks: {}'.format(landmarks[:4]))
        
        The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.
        Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.
LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
        
         image_index = 1
         image = torch.rand([3,64,64]) 
         tuple1=(image, image_index)
         
         Torch.rand:
         Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
         https://pytorch.org/docs/stable/generated/torch.rand.html
         
        __getitem__ to support the indexing such that dataset[idx] can be used to get idx-th sample.
        
        
        You can load all images in the __init__ function, or you can lazily load them in __getitem__.
        
        return------> a tuple: img, label   #https://www.freecodecamp.org/news/python-returns-multiple-values-how-to-return-a-tuple-list-dictionary/

        Labels and the corresponding image paths are saved in labels.csv, their headers are *file* and *label*. 
        img is a tensor???????
        Pil image to tensor:  https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
        
            print("t is: ", t.size()) 
            from torchvision import transforms
            img = transforms.ToPILImage()(t).convert("RGB")
            display(img)
            print(img)
            print(img.size)
        
        
        
        
        def person():
            return "bob", 32, "boston"
   
        https://stackoverflow.com/questions/62660486/using-image-label-dataset-take2-returns-two-tuples-instead-of-a-single-one          
        
        """
        
        

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

"""


models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
    
class LinearClassifier(torch.nn.Module):
    def __init__(self):
    def forward(self, x):
    
class MLPClassifier(torch.nn.Module):
    def __init__(self):
    def forward(self, x):

model_factory = {'linear': LinearClassifier,    'mlp': MLPClassifier, }

def save_model(model)

def load_model(model)

def LossFunction(Y_hat_Vector, y_vector):
  
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

"""

  

def LossFunction (prediction_logit, y_vector):
  Y_hat_Vector = 1/(1+(-prediction_logit).exp()) 
  return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

class ClassificationLoss(torch.nn.Module):
    
  def forward(Y_hat_Vector, y_vector):   #OLD: def forward(self, Y_hat_Vector, y_vector):
        
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      
      #This is the negative log likelihood for logistic regression, need softmax instead.
      #In Logistic Regression, Y_hat_vector is a prediction for ALL x(i) in the data set, so it returns
      #a vector of "i" scalars.  In Softmax, this would return a vector (tensor) of "i" vectors - not scalars).
        
        
"""
        
        Implement the log-likelihood of a softmax classifier.

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')



        Hint: Don't be too fancy, this is a one-liner
"""
         
             
        #raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim):        #input_dim parameter not needed for homework!
      
    super().__init__()
    self.w = Parameter(torch.zeros(input_dim))
    self.b = Parameter(-torch.zeros(1))
    print ("Wandavision, you're inside LinearClassifier class, __init_ constructor, models.py")

  def forward(self, x):      
        
    print ("Wandavision, you're inside LinearClassifier class, forward method, models.py")      
    return (x * self.w[None,:]).sum(dim=1) + self.b 

      
        
"""
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
"""
        #raise NotImplementedError('LinearClassifier.__init__')        
        #raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):

  def __init__(self):
    super().__init__()
    
  #def forward(self, x):

model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}

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
    
    image_dataSET = My_DataSet.get_item(2)
    
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

     
      model_loss.backward()
    
      for p in linear_Classifier_model.parameters():                                       
    
        p.data[:] -= 0.5 * p.grad                    
        p.grad.zero_()


    print (f'*********The local random image:  {image}, the local tuple {tuple1}')
    
    print (f'22222222222222222---->The  image DATASET tuple {image_dataSET}')
    
    fake_Image = My_DataSet.get_fake_image(1)
    real_Image = My_DataSet.get_real_image(1)
    

    print (f'33333333333333333333---->A fake image {fake_Image}')
    print (f'44444444444444444444---->A real image {real_Image}')
    
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
