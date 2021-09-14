import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Image_Transformer
import csv     

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
input_dim = 3*64*64
hidden_size=5

class SuperTuxDataset(Dataset):

  def __init__(self, dataset_path):
  
  
    self.BatchSize = batch_size
    self.imageDATASET = torch.rand([self.BatchSize,3,64,64])   #COMPLETE BUT RANDOM DATA SET
 
    #self.labels = torch.randint(0, 5, (self.BatchSize, ))  
    self.labels = torch.tensor([3, 1, 0, 4, 4, 2, 2, 0, 3, 1, 3, 2, 4, 4, 4, 4, 2, 0, 4, 3, 1, 3, 4, 0,
        3, 1, 2, 4, 4, 1, 4, 3, 3, 0, 0, 3, 4, 4, 1, 2, 2, 2, 4, 1, 0, 1, 1, 4,
        4, 2, 1, 3, 0, 1, 0, 2, 0, 1, 1, 4, 0, 1, 0, 3, 4, 2, 2, 0, 0, 0, 2, 2,
        2, 3, 1, 4, 4, 4, 4, 0, 3, 4, 3, 4, 3, 3, 0, 2, 2, 4, 0, 0, 1, 1, 4, 2,
        2, 2, 0, 3, 3, 4, 1, 1, 0, 2, 1, 1, 2, 3, 0, 2, 0, 4, 4, 1, 1, 0, 2, 0,
        3, 4, 3, 2, 0, 0, 0, 2])
 
    
    file_names = ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg', '00005.jpg', '00006.jpg', '00007.jpg', '00008.jpg', '00009.jpg', '00010.jpg']
    
    self.one_image = Image.open(file_names[0])
    
    image_index = 0
    
    for this_image in file_names:
      self.one_image = Image.open(this_image)
      self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
      self.Image_tensor = self.Image_To_Tensor(self.one_image)
      self.imageDATASET[image_index] = self.Image_tensor
      #self.one_image.show()
      image_index += 1   
  
    #Now just open labels.cv, loop until the file ends, populate file_names, the iterate over file names.
    #OR, iterate through labels until it ends, grab the file name, open, convert, etc.  

    print(f'This is the FAKE RAND DATA SET TENSORS:  {self.imageDATASET[0:, ]}')
  
    print(f'This is the tensor conversion of an actual REAL image:  {self.Image_tensor}')
   
    print(f'I am in the constructor for SuperTuxt, just assigned image tensor above to DATASET zero, this is dataset 0: {self.imageDATASET[0]}')
    
    print (f'These are the fake labels, {self.labels}')
    print (f'These are the file names  {file_names}')
    
    
    val = input("PRESS ANY KEY AND BE BOLD")
    print(val) 
    
    print(f'This is the second fake image tensor in RAND DATASET:  {self.imageDATASET[1]}')
    
    print ("Finally, I will now iterate over labels.cvs file.........")
    
    val = input("PRESS ANY KEY to iteraTE over labels.csv and load ALL DATA")
    print(val) 
    
    
    image_index = 0
  
    with open('labels.csv', newline='') as csvfile:
    
      ImageReader = csv.reader(csvfile) 
      #ImageReader = csv.reader(csvfile, delimiter=' ', quotechar='|') 
      #https://www.geeksforgeeks.org/working-csv-files-python/     
      
      for row in ImageReader:
        
        #print(', '.join(row))
        #print (f'File names ONLY-------------------{row[0]}')
        #print (f'Label names only=================={row[1]}')
        
        if image_index > 0:
          
          image_file_name = "..\data\\train\\"+row[0] 
          print(image_file_name)
          self.one_image = Image.open(image_file_name)
          self.Image_To_Tensor = Image_Transformer.transforms.ToTensor()
          self.Image_tensor = self.Image_To_Tensor(self.one_image)
          self.imageDATASET[image_index] = self.Image_tensor
          
        image_index += 1 
        if image_index == self.BatchSize:
            break
    

    
    val = input("There!  I just LOADED ALL DATA up to Batch size!")
    print(val) 
    
         
        
  def __len__(self):
    return (self.imageDATASET.size(0))  
         
  def __getitem__(self, idx):     
    return (self.imageDATASET[idx], self.labels[idx]) #image Tensor size (3,64,64) range [0,1], label is int.
    
        

def load_data(dataset_path, num_workers=0, batch_size=128):     #use this in train.py
    #all this does is return the data_loader to use in SGD?
  
    #https://machinelearningknowledge.ai/pytorch-dataloader-tutorial-with-example/
    
    dataset = SuperTuxDataset(dataset_path)   #In Orginal
        
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)  
    
    
def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

"""
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py models.py 
"""

def LossFunction (prediction_logit, y_vector):      #FOR TESTING UNIT CIRCLE ERASE
  
  Y_hat_Vector = 1/(1+(-prediction_logit).exp())   #Take the sigmoid of the logit  
  return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
 
      

class ClassificationLoss(torch.nn.Module):
  
    
    #**You should implement the log-likelihood of a softmax classifier.
    #https://pytorch.org/docs/master/nn.html#torch.nn.LogSoftmax
    #https://pytorch.org/docs/master/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss    (NEGATIVE LOSS 
    #LIKELIHOOD use with LOG-softmax.)   
    #https://pytorch.org/docs/master/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax
          
  def forward(self, Y_hat_Vector, y_vector):   #OLD: def forward(self, input, target):
      
    m = nn.LogSoftmax()
    input = torch.randn(2, 3)
    output = m(input)
        
    return -(y_vector.float() * (Y_hat_Vector+1e-10).log() +(1-y_vector.float()) * (1-Y_hat_Vector+1e-10).log() ).mean()
      
    #This is the negative log likelihood for logistic regression, need SOFTMAX instead.
    #In Logistic Regression, Y_hat_vector is a prediction for ALL x(i) in the data set, so it returns
    #a vector of i scalars.  In Softmax, this would return a vector (tensor) of i vectors - not scalars).
      
      
      
        
class LinearClassifier(torch.nn.Module):

  def __init__(self, input_dim):        #input_dim parameter not needed for homework, I added thiS!
      
    super().__init__()   #original
       
    
    self.w = Parameter(torch.zeros(input_dim))  #added
    self.b = Parameter(-torch.zeros(1))         #added
    
   
  def forward(self, x):      
    
    return (x * self.w[None,:]).sum(dim=1) + self.b 
   


#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP*************
#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP*************
#*************MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP*************



class MLPClassifier(torch.nn.Module):  

  def __init__(self):   
   
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
                
  def forward(self, flat_image):   
    return self.network(flat_image)
  
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


model_factory = { 'linear': LinearClassifier, 'mlp': MLPClassifier, }
  
  
  

#TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN 
#TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN BEGIN TRAIN 

n_epochs = 10                   #CHANGE EPOCHS to 100 !!!!!!!!
batch_size = 128
input_size = 64*64*3

def train(args):

    
    image_index = 1                   #test code
    local_fake_image = torch.rand([3,64,64])    #test code
    Local_Fake_Image_Set = torch.rand([batch_size,3,64,64])  #no labels
    local_tuple=(local_fake_image, LABEL_NAMES[image_index])       #test code, but the second tuple should be int.
    
    Batch_Size_Sixtensors = torch.zeros(batch_size,6)  #test code to test output from network
       
 
    #y_labels_tensor = torch.ones(batch_size,6)  #Actual Y to compute softmax loss, [0,1,0,0,0,0], NEEDS pre-processing
    #y_labels_tensor = (torch.rand(size=(batch_size,6)) < 0.25).int()  #Actual Y to compute softmax loss, e.g,  [0,1,0,0,0,0]
    
    rand_mat = torch.rand(batch_size, 6)
    k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
    
    
    #Define the one-hot-encoded Y vectors.  Y_labels is Y, 
    #I am not sure if we only convert to 6-tensor at the end.  Presumably
    #Since the output is a 6 tensor - all values have to be updated and the proper value softmaxED
    
    y_labels_tensor = torch.where((rand_mat <= k_th_quant),torch.tensor(1),torch.tensor(0))
    y_hat_tensor = torch.rand(batch_size,6)  #this is going to change when put through network
    
      
    
    My_DataSet = SuperTuxDataset('c:\fakepath')      
    #This should load the data un the constructor using load_data function    
    #Sept 13, 2021: This will load a random, fake data set, with the first entry 0 a real image
        
        
    real_Image, an_int_label_from_dataset = My_DataSet[0]   #gets the real image TENSOR  kel76y
    Second_Fake_image_tuple = My_DataSet[2]                 #this should get fake image #2    
    
        
    
     #SOME FAKE TEST DATA FOR UNIT CIRCLE SGD
    
    x = torch.rand([10,2])          #for testing unit circle SGD
    true_y = ((x**2).sum(1) < 1)    #for testing unit circle SGD

    
    
#LOAD DATA LOAD LOAD DATA LOAD DATA LOAD DATA LOAD DATA LOAD DATA LOAD DATA LOAD DATA LOAD DATA LOAD DATA LOAD DATA 
#Use load_data() function, somethhing like my_loader = load_data("path"). 
#https://machinelearningknowledge.ai/pytorch-dataloader-tutorial-with-example/
#trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    
    #create the network
    linear_Classifier_model = model_factory[args.model](2)     #LINEAR CLASSIFIER BY DEFAULT IN THE COMMAND LINE, USED FOR GRADING
    MLPx = MLPClassifier()                                     #MLP Used for Testing, Erase - use command line args to call this
    
    #create the optimizer for MLP (change to use args)
    optimizer = torch.optim.SGD(MLPx.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)        
    #just defining the optimizer, call it w/ step to update weights 
    #but must implement loss.backward first to get the gradients
    
    # Create the loss - For these batch_size images, the network predicted the labels as set forth by y_hat_vector 
    #(change to tensor!!!)
    
    loss_object = ClassificationLoss()   
    #call it with Y_hat_Vector, y_vector... input (y_hat predicted labels) and 
    #target (y_vector actual image label)
    #now I can call it as such, model_loss = loss_object (y_hat_tensor, y_tensor) where y_tensor has actual values.. 
    #or is it tuples???  
    

    #BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD BEGIN SGD 
    
    global_step = 0
    
    
    #*******************************************BEGIN  UNIT CIRCLE TRAINING*********************************************
    #*******************************************BEGIN  UNIT CIRCLE TRAINING*********************************************
    
    print ("--------------------------------STARTING TRAINING---------------------------------")
    
    for epoch in range(n_epochs): 
    
    
      # Shuffle the data
      permutation = torch.randperm(Local_Fake_Image_Set.size(0))  #generate batch_size numbers from 0 to batch_size-1
      
      
    
    
      #iteration of unit circle
      Y_hat = linear_Classifier_model.forward(x)    # unit circle erase
      model_loss = LossFunction(Y_hat, true_y)     #unit circle
      print ("UNIT CIRCLE Model LOSS:", model_loss) #unit circle
      print ("UNIT CIRCLE Truth Value of Logits", Y_hat > .5 )  #unit circle
      print (f'UNIT CIRCLE model loss at epoch {epoch} is {model_loss} and the prediction y_hat is {Y_hat}, while the y is {true_y}')

      #model_loss = ClassificationLoss.forward(prediction_logit, true_y)
      
      #Y_hat_sigmoid_of_logit = 1/(1+(-prediction_logit).exp())  
      
           
      model_loss.backward()   #get the gradients with the computation graph
    
      train_accuracy = []
      
      for p in linear_Classifier_model.parameters():         #update parameters, replace with optimizer step                
    
        p.data[:] -= 0.5 * p.grad                    
        p.grad.zero_()

#*********************************************************END TRAINING*************************************************************
#*********************************************************END TRAINING*************************************************************
#*********************************************************END TRAINING*************************************************************
#*********************************************************END TRAINING*************************************************************


    print ("--------------------------------FINISHED TRAINING---------------------------------")
    
    print (f'*********A sample fake image created with tensor RAND, this is NOT from the Dataset!:  {local_fake_image}, the local tuple {local_tuple}')
    
    
    print (f'22222222222222222---->Item 2 FROM THE OBJECT DATASET, which was created w/ RAND, this SHOULD BE A TUPLE: {Second_Fake_image_tuple}')
    
    print (f'One image from Second_Fake_image_tuple, which I got from getitem method {Second_Fake_image_tuple[0]}')
    
    print (f'Here is the Local_Fake_Image_Set.size, {Local_Fake_Image_Set.size(0)}  and  the permutation data set {[permutation]}')
    
    print (f'Here is the permutation iterative which goes in the for loop, len(permutation)-batch_size+1 = {len(permutation)} minus {batch_size+1}')
    
    val = input("Enter your value: ")
    print(val)
    
    print (f'Just did {n_epochs} Gradient Descent iterations with ten UNIT CIRCLE points ONLY!!!')
    
    
    #GRADIENT DESCENT USING THE  IMAGES----------------------------------------------------
    #LOAD THE 250 IMAGES 
    
    
    print (f'HERE I GO - ABOUT TO CREATE A MLP.................., these are the image dimensions:  {Second_Fake_image_tuple[0].size()}')
    
    val = input("Enter your value, then I will print the flattened image: ")
    print(val)

    
    
    flatened_Image = real_Image.view(real_Image.size(0), -1).view(-1)    #kel76y
    
    
    print ("*****************I AM IN THE TRAIN MODULE NOW******************************************")
    print (f'This is the flattened image {flatened_Image}, of size {flatened_Image.size()}, , original tensor size was {real_Image[0].size()}')
    
    print (f'This is the Zero-th 6-tensor Before taking the output from my neural network: {Batch_Size_Sixtensors[0]}')
    
    print (f'This is the real image, which should be 3x64x64: {real_Image[0]}')
    
    
    
    
     
    val = input("Just printed the flattened image (before above 6-tensor).  Enter your value: ")
    print(val)
    
    Batch_Size_Sixtensors[0] = MLPx(flatened_Image)  #kel76y
    
    print (f'Here is the zero-th 6-tensor AFTER putting it through the MLPx Network: {Batch_Size_Sixtensors[0]}')
    
    print(f'Here is the 7th and 110th fake images {Local_Fake_Image_Set[7]}, {Local_Fake_Image_Set[110]}')
    
    
    val = input("Just printed The 7th and 110th fake images.  Press enter to see y_labels_tensor and y_hat_tensor")
    print(val)
    
    print (f'Thank you, here they are, the y_labels_tensor is {y_labels_tensor}, and the y_hat_tensor is {y_hat_tensor}')
    
    
    print ("Now you need to figure out how to populate y_labels_tensor using labels.csv file")
    
    print (f'real_image {real_Image} should be the same as My_DataSet[0] {My_DataSet[0]}')
    
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
