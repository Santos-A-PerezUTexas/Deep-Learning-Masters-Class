
  #HOMEWORK 3
  #Oct 13, 2021
  #no pooling
  #Always pad by kernel_size / 2, use an odd kernel_size
  #Oct 13:  DO I HAVE TO use transforms on the labels?
  #Oct 13 - does randcrop 64 do anything?
  #OCT 13 NIGHT:  TOOK OUT BLOCK OF FCN


import torch
import torch.nn.functional as F

"""
1.   TUNE THE CNNClassifier 


  -Input normalization - https://en.wikipedia.org/wiki/Batch_normalization?
  -Residual blocks (see video on residual connections pytorch programming by professor)
  -Dropout
  -Data augmentations (Both geometric and color augmentations are important. Be aggressive here. Different levels of supertux have radically different lighting.)
  -Weight regularization
  -Early stopping


Input Normatlization

  http://www.philkr.net/dl_class/lectures/making_it_work/09.pdf

Augmentation:

  https://drive.google.com/file/d/1XOYF7qxlj3sDPk0dzV5w2jo4THvspQkk/view
  https://discuss.pytorch.org/t/data-augmentation-in-pytorch/7925
  https://www.youtube.com/watch?v=Zvd276j9sZ8



Residual Connections:

  https://drive.google.com/file/d/1in6SpWW0pRCE_aibvG2jCUvphlCt0wdp/view

  http://www.philkr.net/dl_class/lectures/making_it_work/18.html
  (this link also has batch normalization)

  http://www.philkr.net/dl_class/lectures/making_it_work/17.pdf

Identity Mapping for Residual Connections:

  https://openreview.net/pdf?id=ryxB0Rtxx
  reparameterization of the convolutional layers such that when all trainable weights
  are 0, the layer represents the identity function. Formally, for an input x, each 
  residual layer has the form x + h(x), rather than h(x)

"""
class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Compute mean(-log(softmax(input)_label))
        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)
        @return:  torch.Tensor((,))
   """
        return F.cross_entropy(input, target)
        
#########################CNN  BEGIN

class CNNClassifier(torch.nn.Module):

  class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity

##########CNN INIT  (ABOVE IS CNN BLOCK)

  def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case

     
      self.layer1 = torch.nn.Sequential(
      
            torch.nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            self.Block(32,32),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                      )    
            
      self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))                  
        
              
      self.fc1 = torch.nn.Linear(32 * 32 * 32, 6)   
      
      self.drop_out = torch.nn.Dropout()
          
  #######################CNN FORWARD

  def forward(self, images_batch):
   
     
      out = self.layer1(images_batch)
      out = out.reshape(out.size(0), -1)
      out = self.drop_out(out)
      out = self.fc1(out)
      
                   
      return out

######################################### END CNN

#########################################BEGIN FCN

class FCN(torch.nn.Module): 
  class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=3/2, stride=1, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=3/2, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1),
                                                      torch.nn.BatchNorm2d(n_output))
        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity


  def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case

     
      self.layer1 = torch.nn.Sequential(
      
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=5/2),
            torch.nn.ReLU(),
            #self.Block(64,128),
            torch.nn.BatchNorm2d(128),   #image is now 65*65
            
                                     )    
        
      self.layer2 = torch.nn.Sequential( 
        
        #https://piazza.com/class/ksjhagmd59d6sg?cid=342
        #https://piazza.com/class/ksjhagmd59d6sg?cid=342
        #https://piazza.com/class/ksjhagmd59d6sg?cid=342
        
        #torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        #torch.nn.BatchNorm2d(512),
        #torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        #torch.nn.BatchNorm2d(256),
        #torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        #torch.nn.BatchNorm2d(128),
        
        #Oct 13 Night: image is 65x65, downsample to 64x64

        torch.nn.ConvTranspose2d(128, 32, kernel_size=(2,2), stride=(1,1), padding=(1,1), dilation=1, output_padding=(0,0)),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 5, kernel_size=1)


        #torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3/2, dilation=1, output_padding=1),
        #torch.nn.BatchNorm2d(64),
        #torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=3/2, dilation=1, output_padding=1),
        #torch.nn.BatchNorm2d(32),
        
                      )
  def forward(self, images_batch):
   
     
      out = self.layer1(images_batch)
      out = self.layer2(out)
      
      #NOW DOWNSAMPLING!
      #out = out.reshape(out.size(0), -1)
      #out = self.drop_out(out)
      #out = self.fc1(out)
      
                   
      return out

        #Oct 13, 2021:  https://medium.com/swlh/how-data-augmentation-improves-your-cnn-performance-an-experiment-in-pytorch-and-torchvision-e5fb36d038fb

        #Simply, our goal is to take either a RGB color image (height×width×3) or a grayscale
        # image (height×width×1) and output a segmentation map where each pixel contains a
        #class label represented as an integer (height×width×1).
        
        #LOSS:   pixel-wide cross entropy loss, see https://www.jeremyjordan.me/semantic-segmentation/
        #Oct 10, 2021 (above)

        #LOSS: IOU!!!!!!!!!!!!!!!! https://www.jeremyjordan.me/evaluating-image-segmentation-models/

                     #PIXEL WIDE CROSS ENTROPY LOSS
        #https://discuss.pytorch.org/t/unet-pixel-wise-weighted-loss-function/46689
        #https://stackoverflow.com/questions/50896412/channel-wise-crossentropyloss-for-image-segmentation-in-pytorch
        #the documentation on the torch.nn.CrossEntropy() function can be found
        # here and the code can be found here. The built-in functions do indeed already support
        # KD cross-entropy loss.
        

        #Oct 10, 2021 https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
        #class NLLLoss(_WeightedLoss):
        
        #Oct 10, 2021:  CROSS ENTROPY LOSS:  https://discuss.pytorch.org/t/cross-entropy-loss-error-on-image-segmentation/60194
        #nn.CrossEntropyLoss is usually applied for multi class classification/segmentation
        # use cases, where you are dealing with more than two classes.
        #In this case, your target should be a LongTensor, should not have the channel dimension, and should contain the class indices in [0, nb_classes-1].


        #Oct 10, 2021: https://www.jeremyjordan.me/semantic-segmentation/
        #Oct 10, 2021: https://www.jeremyjordan.me/semantic-segmentation/                                   
        #Oct 10, 2021:  https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/
        #Oct 10, 2021: https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-with-pytorch/blob/master/NET_FCN.py

        #https://github.com/pochih/FCN-pytorch/tree/master/python
        #https://github.com/wkentaro/pytorch-fcn/tree/master/torchfcn
        #https://medium.com/@iceberg12/semantic-segmentation-applied-on-car-road-4ee62622292f
        
        #https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
        #https://discuss.pytorch.org/t/add-residual-connection/20148/6
        #https://stats.stackexchange.com/questions/321054/what-are-residual-connections-in-rnns
        
        #Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        #Hint: Use up-convolutions
        #Hint: Use skip connections
        #Hint: Use residual connections
        #Hint: Always pad by kernel_size / 2, use an odd kernel_size
        
      
      #raise NotImplementedError('FCN.__init__')



        
        #@x: torch.Tensor((B,3,H,W))
        #@return: torch.Tensor((B,6,H,W))<--- returns 5 channels of HW instead!!!
        #Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        #Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
        #     if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
        #     convolution
       
        


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
