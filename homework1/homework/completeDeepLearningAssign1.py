import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Image_Transformer
import csv     #utils.py

iterations_for_sgd = 10
#http://www.philkr.net/dl_class/lectures/deep_networks/10.html
#http://www.philkr.net/dl_class/lectures/deep_networks/10.html    FOLLOW THIS CODE!

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
img = transforms.ToPILImage()(t).convert("RGB")

Hint: You have (at least) two options on how to load the dataset. 
You can load all images in the __init__ function, or you can lazily load them in __getitem__.
If you load all images in __init__, make sure you convert the image to a tensor in the constructor, otherwise, you might get an OSError: [Errno 24] Too many open files.


"""


LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
input_dim = 3*64*64
hidden_size=5

class SuperTuxDataset(Dataset):

  def __init__(self, dataset_path):
  
    self.BatchSize = 20
    self.X_imageDATASET = torch.zeros([self.BatchSize,3,64,64]) 
    self.size = 64,64
    self.one_image = Image.open(r"sample_image.jpg")
    
    print(f'Behold, the EMPTY DATASET:  {self.X_imageDATASET[0:, ]}') 
    
    #convert image to tensor 
     
    self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
    
    self.Image_tensor = self.Image_To_Tensor(self.one_image)
    self.X_imageDATASET[0] = self.Image_To_Tensor(self.one_image)
    
    
    print(f'Behold, the image tensor:  {self.Image_tensor}') 
    
    #print(f'Behold, the image tensor in DATASET:  {self.X_imageDATASET[0:, ]}') shows entire tensor
    print(f'Behold, the image tensor in DATASET:  {self.X_imageDATASET[0]}')
    
    print ("Just opened the sample image, about to show it to you.")
    self.one_image.show()
    
     
    #Hint: Use torchvision.transforms.ToTensor() to convert the PIL image to a pytorch tensor.
    #img = transforms.ToPILImage()(t).convert("RGB")
    
      
  
    #LOAD THE DATA-----------------------------------------
    
    #Can't use this because load_data makes a new Dataset, calls the Constructor, and recurses!
    #temp_data = load_data('../data/train', num_workers=0, batch_size=128)
    
    #convert to tensor and return tensor
    
        
  def __len__(self):
    return (3000)  
         
  def __getitem__(self, idx):     
    return (self.X_imageDATASET[idx], LABEL_NAMES[idx])
  
  def get_item(self, idx):     
    return(self.__getitem__(idx))
    
  def get_fake_image(self, idx):     
    return(self.X_imageDATASET[idx], LABEL_NAMES[idx])

  def get_real_image(self, idx):     
    return (self.X_imageDATASET[idx], LABEL_NAMES[idx])
    
        
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
        
        

def load_data(dataset_path, num_workers=0, batch_size=128):     #use this in train.py
    
    dataset = SuperTuxDataset(dataset_path)   #In Orginal
    
    Y_labels = full_set_accuracy = [0] * batch_size  #I added this
    
    #https://medium.com/bivek-adhikari/creating-custom-datasets-and-dataloaders-with-pytorch-7e9d2f06b660
    #https://medium.com/bivek-adhikari/creating-custom-datasets-and-dataloaders-with-pytorch-7e9d2f06b660
    
    #https://docs.python.org/3/library/csv.html#module-csv
    #STDataset = pd.read_csv('data/labels.csv')  #pandas, can't use
    
    with open('labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      #cvs.reader returns a reader object which will iterate over lines in the given csvfile
    
      for row in ImageReader:
        print(', '.join(row))
        print (f'ROW IS -------------------{row}')
    
      #for row in range(1, 250):
        #print(', '.join(row))
    
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)  #original
    #https://pytorch.org/docs/stable/data.html  ,,,, defines DataLoader
    
    
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
    #https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
    
    class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(1,1)
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x) # instead of Heaviside step fn
        return output
        
model_factory = {'linear': LinearClassifier,    'mlp': MLPClassifier, }

def save_model(model)

def load_model(model)

def LossFunction(Y_hat_Vector, y_vector):
  
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

"""

  

def LossFunction (prediction_logit, y_vector):
 
 #this is not part of the original???
 
  Y_hat_Vector = 1/(1+(-prediction_logit).exp())   #Take the sigmoid of the logit
  
  return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      

class ClassificationLoss(torch.nn.Module):
  
    #https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/loss.py
    
    #**You should implement the log-likelihood of a softmax classifier.
    #https://pytorch.org/docs/master/nn.html#torch.nn.LogSoftmax
    #https://pytorch.org/docs/master/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss    (NEGATIVE LOSS 
    #LIKELIHOOD use with LOG-softmax.)
    
    
    #https://pytorch.org/docs/master/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
      
    #https://pytorch.org/docs/master/nn.html#torch.nn.LogSoftmax
    #https://discuss.pytorch.org/t/does-nllloss-handle-log-softmax-and-softmax-in-the-same-way/8835
    #https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816
    #https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/other/pytorch-lossfunc-cheatsheet.md
    
    #If you apply Pytorch's CrossEntropyLoss to your output layer
    #you get the same result as applying Pytorch's NLLLoss to a
    #LogSoftmax layer added after your original output layer.
    
    #torch.nn.functional.binary_cross_entropy takes logistic sigmoid values as inputs
    #torch.nn.functional.binary_cross_entropy_with_logits takes logits as inputs
    #torch.nn.functional.cross_entropy takes logits as inputs (performs log_softmax internally)
    #NEED THIS***********torch.nn.functional.nll_loss is like cross_entropy but takes log-probabilities (log-softmax) values as inputs
       
       
  
    
  def forward(self, Y_hat_Vector, y_vector):   #OLD: def forward(self, input, target):
  
    #Why forward method:  https://discuss.pytorch.org/t/about-the-nn-module-forward/20858
 
    m = nn.LogSoftmax()
    input = torch.randn(2, 3)
    output = m(input)
        
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      
    #This is the negative log likelihood for logistic regression, need SOFTMAX instead.
    #In Logistic Regression, Y_hat_vector is a prediction for ALL x(i) in the data set, so it returns
    #a vector of i scalars.  In Softmax, this would return a vector (tensor) of i vectors - not scalars).
        
        
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

#multinomonial logistic regression  - NO SIGMOID, SOFTMAX LAYER INSTEAD
#Your forward function receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor (one value per class).
#B is the batch size, it's a hyper parameter we can set.

  def __init__(self, input_dim):        #input_dim parameter not needed for homework, I added thiS!
      
    super().__init__()   #original
    
    #https://www.programcreek.com/python/example/107699/torch.nn.Linear
    #https://www.programcreek.com/python/example/107699/torch.nn.Linear
    
    #self.linear = torch.nn.Linear(4096, 6)  
    
    
    self.w = Parameter(torch.zeros(input_dim))  #added
    self.b = Parameter(-torch.zeros(1))         #added
    
    
                
    print ("Wandavision, you're inside LinearClassifier class, __init_ constructor, models.py")

  def forward(self, x):      
    
    #DOES NOT USE SIGMOID
    #DOES NOT USE SIGMOID
    #DOES NOT USE SIGMOID  CONFIRMED.  (MLP Uses Relu).
    
    # x is a (B,3,64,64) tensor, so x[i] is one image
    #x: torch.Tensor((B,3,64,64))
    #return: torch.Tensor((B,6))
        
    #Multinomial Logistic Regression?  https://aaronkub.com/2020/02/12/logistic-regression-with-pytorch.html
    #https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
    #torch.nn.Linear(input_dim, output_dim)
    #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
       
    print ("Wandavision, you're inside LinearClassifier class, forward method, models.py")      
    return (x * self.w[None,:]).sum(dim=1) + self.b 
   
"""
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
"""
        #raise NotImplementedError('LinearClassifier.__init__')        
        #raise NotImplementedError('LinearClassifier.forward')


#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP 
#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP 
#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP 
#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP 
#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP 

class MLPClassifier(torch.nn.Module):  #***************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP 

  #Your forward function receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor (one value per class).
  #Two layers are sufficient.  TWO LAYERS - OUTPUT LAYER, AND INPUT LAYER?  IS RELU A LAYER?, so INPUT L, RELU L
  #Keep the first layer small to save parameters. (12K bits??? small?)
  #PROFESSOR: The inputs and OUTPUTS to the multi-layer perceptron ARE THE SAME as the linear classifier.
  #Use ReLU layers as non-linearities.  Just "Add" a Relu Layer ONE LINE.
  #Use ReLU layers as non-linearities.  (USE  SIGMOID OR SOFTMAX GOOD ENOUGH????)
  #PER PROF ITS THE SAME, BUT W/ RELU (I GUESS ALSO SOFTMAX!)
  #Might require some tuning of your training code. Try to move most modifications to command-line arguments 
  #in  ArgumentParser

  def __init__(self):   #I added input_dim, not in original   MLP CONSTRUCTOR MLP CONSTRUCTOR MLP CONSTRUCTOR MLP CONSTRUCTOR MLP CONSTRUCTOR MLP CONSTRUCTOR 
   
    super().__init__()
    
     #flatten t0 12K features (input_dim = 3*64*64)
     #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
     #https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-scratch.html  READ THIS!!!!!!
     #https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/5
     #model.eval()
     #output = net(input)
     #sm = torch.nn.Softmax()
     #probabilities = sm(output)
     #print(probabilities )
     
     
    print(f'The input_dimension is --------------------->{input_dim}')
    print(f'The hidden size is --------------------->{hidden_size}')
     
    self.layer1=torch.nn.Linear(input_dim, hidden_size)
    self.REluLayer =  torch.nn.ReLU(inplace=False)
    self.layer2=torch.nn.Linear(hidden_size, 6)
     
    self.network = torch.nn.Sequential( 
                torch.nn.Linear(input_dim, hidden_size),   #keep this small???
                torch.nn.ReLU(inplace=False),                                               #THIS IS FOR THE MLP!!!
                torch.nn.Linear(hidden_size, 6)  
                )
                
                #this later has 6 neurons, fed to softmax
                #nn.CrossEntropyLoss()
                #nn.LogSoftmax(), nn.NLLLoss()  -----can use cross-entropy
                
  def forward(self, flat_image):   
    return self.network(flat_image)
  
  #def forward(self, multiple_image_tensor):   
  #receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor
         
  #  return self.network(multiple_image_tensor.view(multiple_image_tensor.size(0), -1))
    #    return self.network(multiple_image_tensor.view(multiple_image_tensor.size(0), -1)).view(-1)
  
  
  
                
     #SEPTEMBER 6, 2021
     #https://www.machinecurve.com/index.php/2021/01/26/creating-a-multilayer-perceptron-with-pytorch-and-lightning/
     #https://majianglin2003.medium.com/create-neural-network-with-pytorch-1f91054fe229
     
     
""" nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))                                         

     def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):
    #Forward pass
    return self.layers(x)    """
                
           
   #https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
      
  #    class Perceptron(torch.nn.Module):
  #   def __init__(self):
  #      super(Perceptron, self).__init__()
  #      self.fc = nn.Linear(1,1)
  #      self.relu = torch.nn.ReLU() # instead of Heaviside step fn
  #   
  #    def forward(self, x):
  #       output = self.fc(x)
  #       output = self.relu(x) # instead of Heaviside step fn
  #       return output



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


model_factory = { 'linear': LinearClassifier, 'mlp': MLPClassifier, }
  
  
  
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
    
    
    My_Real_DataSet = load_data('../data/train', num_workers=0, batch_size=128)
    
    
    image_dataSET = My_DataSet.get_item(2)
    
    linear_Classifier_model = model_factory[args.model](2)     #DEFINING THE CLASSIFIER HERE
    
      #defaults to linear
      #OR   linear_Classifier_model = LinearClassifier(2)
      #dimension of weights is 2 just example...
    
    #raise NotImplementedError('train')
     
    x = torch.rand([10,2]) 
    true_y = ((x**2).sum(1) < 1)
    
    #iterations_for_sgd defined at the beggining
    for iteration in range(iterations_for_sgd): 
    
      Y_hat = linear_Classifier_model.forward(x)
      
      #model_loss = ClassificationLoss.forward(prediction_logit, true_y)
      
      #Y_hat_sigmoid_of_logit = 1/(1+(-prediction_logit).exp())  
      
      model_loss = LossFunction(Y_hat, true_y)
      
      print ("Model LOSS:", model_loss)
      print ("Truth Value of Logits", Y_hat > .5 )

      print (f'model loss at iteration {iteration} is {model_loss} and the prediction y_hat is {Y_hat}, while the y is {true_y}')

     
      model_loss.backward()   #get the gradients with the computation graph
    
      for p in linear_Classifier_model.parameters():                       #update parameters                
    
        p.data[:] -= 0.5 * p.grad                    
        p.grad.zero_()


    print (f'*********The local random image:  {image}, the local tuple {tuple1}')
    
    print (f'22222222222222222---->The  image DATASET tuple {image_dataSET}')
    
    val = input("Enter your value: ")
    print(val)

    fake_Image = My_DataSet.get_fake_image(0)
    real_Image = My_DataSet.get_real_image(0)
    

    print (f'33333333333333333333---->A fake image {fake_Image}')   #Image with Label
    
    val = input("Enter your value: ")
    print(val)

    print (f'44444444444444444444---->A real image {real_Image[0]}')  #Image without label
    
    print (f'Just did {iterations_for_sgd} Gradient Descent iterations with ten UNIT CIRCLE points ONLY!!!')
    
    
    #GRADIENT DESCENT USING THE  IMAGES----------------------------------------------------
    #LOAD THE 250 IMAGES 
    
    
    print (f'HERE I GO - ABOUT TO CREATE A MLP.................., these are the image dimensions:  {real_Image[0].size()}')
    
    val = input("Enter your value, then I will print th flattened image: ")
    print(val)

    MLPx = MLPClassifier()
    
    empty_tensor = torch.zeros(6)
    
    flatened_Image = real_Image[0].view(real_Image[0].size(0), -1).view(-1)
    
    
    print (f'This is the flattened image {flatened_Image}')
    print (f'This is the Zero 6-tensor Before taking the output from my neural network: {empty_tensor}')
    
     
    val = input("Just printed the flattaned image.  Enter your value: ")
    print(val)
    
    empty_tensor = MLPx(flatened_Image)
    
    print (f'You rock man.  Here is the 6-tensor you stud: {empty_tensor}')
    
    
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
