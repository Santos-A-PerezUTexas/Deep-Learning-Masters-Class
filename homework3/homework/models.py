
  #HOMEWORK 3
  #Oct 13, 2021
  #no pooling
  #Always pad by kernel_size / 2, use an odd kernel_size
  #Oct 13:  DO I HAVE TO use transforms on the labels?
  #Oct 13 - does randcrop 64 do anything?


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
          
    
  def forward(self, images_batch):
   
     
      out = self.layer1(images_batch)
      out = out.reshape(out.size(0), -1)
      out = self.drop_out(out)
      out = self.fc1(out)
      
                   
      return out


class FCN(torch.nn.Module):
  #def __init__(self, pretrained_net, n_class):
  def __init__(self):

    super().__init__()
    
    #self.cnn = CNNClassifier()
    self.cnn = torch.nn.Sequential(
      
            torch.nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=7/2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=3/2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                      )  
    self.n_class = 5
    #self.pretrained_net = pretrained_net
    self.relu    = torch.nn.ReLU(inplace=True)
    self.deconv1 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn1     = torch.nn.BatchNorm2d(512)
    self.deconv2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn2     = torch.nn.BatchNorm2d(256)
    self.deconv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn3     = torch.nn.BatchNorm2d(128)
    self.deconv4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn4     = torch.nn.BatchNorm2d(64)
    self.deconv5 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
    self.bn5     = torch.nn.BatchNorm2d(32)
    self.classifier = torch.nn.Conv2d(32, self.n_class, kernel_size=1)
  
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

  def forward(self, x):

        output = self.cnn(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))<--- returns 5 channels of HW instead!!!
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        


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
