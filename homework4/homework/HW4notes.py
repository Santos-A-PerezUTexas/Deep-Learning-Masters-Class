
https://piazza.com/class/ksjhagmd59d6sg?cid=672
  
  
SLIDES: https://piazza.com/class/ksjhagmd59d6sg?cid=191
SLIDES: https://piazza.com/class/ksjhagmd59d6sg?cid=191
SLIDES: https://piazza.com/class/ksjhagmd59d6sg?cid=191
kel76y  ---> Defining Heatmaps
Example Output:  https://piazza.com/class/ksjhagmd59d6sg?cid=610
                
                
HW4 SLIDES:  https://docs.google.com/presentation/d/e/2PACX-1vR6bYbuIJeA1og36QOdEMANAXbdbCpjaPVWkZNthAqIJ-iKOPKtYcLnb7n_rlNGvbiukP7W3JLaM1HQ/pub?start=false&loop=false&delayms=3000&slide=id.p1
        
    



Model
How much should our model differ from our answer in HW3? I was able to get full marks on HW3 with the model I've implemented
but now it performs poorly with AP  less than 0.2. I have also implemented Focal loss and have attempted to train with Focal loss and BCEWithLogitsLoss and the
result is the same. Is there any major architecture changes that need to be done in our FCN model?

---->Make sure that your detect function is working correctly. If you look to grader/tests.py, you will see that the tests for the detect function only checks if 
---->the output is in the correct format, not that it actually returns valid detections. See below from line 177.

            assert len(d) == 3, 'Return three lists of detections'
            assert len(d[0]) <= 30 and len(d[1]) <= 30 and len(d[2]) <= 30, 'Returned more than 30 detections per class'
            assert all(len(i) == 5 for c in d for i in c), 'Each detection should be a tuple (score, cx, cy, w/2, h/2)'

------->Your detect function is used by the grader when it evaluates your actual model at line 197. So, if your detect function is working properly, 
-------->you will also see poor performance when the grader evaluates your model.

-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
 
---->Q1:

dense_transform.py, Toheatmap class: this line does not work as it says image is PIL so does not have shape:

  peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)

  After I change it to:
    
  peak, size = detections_to_heatmap(dets, image.size, radius=self.radius)

Is this correct?

----->A:  ToHeatmap should be used after ToTensor, that way it is not a PIL image anymore.
  

-------Q2:

When the model received the input from dataloader, the input is a tuple of 3 tensors of size:

torch.Size([32, 3, 96, 128])    #Image
torch.Size([32, 3, 128, 96])    #Heatmap
torch.Size([32, 2, 128, 96])    #Labels for Extracredit

I understand the first is image. What about the rest two? Are those heatmaps of objects? And why is the 2nd dim different? 

-------->A. The second tensor is the heatmap, the third is the labels for the extra credit (full object detection). 
  They should have the same dimensions as the image; maybe the switch was caused by the edit you made?
  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
  
 
  assertion backwards for the Radius_1 test parameters.
  
          assert len(p) == (img > 0).sum(), 'Expected exactly %d detections, got %d' % (len(p), (img > 0).sum())
    
    STRING SHOULD INSTEAD BE:
      
               'Expected exactly %d detections, got %d' % ((img > 0).sum(),len(p))
 
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
    
      
      Could the local minimum also be an object? eg a bright light from an oncoming vehicle against a dark background?
           --->The peaks are extracted from your network’s output so you don’t have to worry about what the image looks like before.
  
  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
  Width and Height
Is there are a resource we can look into when dealing with 
balancing  the two losses between the image and the width and height channels?
or calculating losses for separate channels?   

----->For detection, BCE loss range from 0~1, for height and width prediction, the loss is much larger.
----->So you may want to assign BCE loss a much higher weight.
    -----> Binary Cross Entropy loss (criterion)  between the target and the input probabilities
    ----->CLASStorch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')  #notice weight parameter
            -> weight (Tensor, optional) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.

I see so would this affect the Focal Loss implementation or should we stick to BCE for height and widht?
How would we go about doing the backwards  pass during training when the losses have been calculated?
---->For height and width, as a regression task, you need MSE or MAE loss function. 
---->For backpropagation, just like:
  
              L1 = BCE()
              L2 = MSE()

              L = w1*L1 + w2*L2
              L.backward()
                
-------------------------------------------------------------------------------------------------------------------------------
                        LOSS FUNCTION - TWO OF THEM.  MSE() and BCE()  (FOCAL LOSS?)   KEYWORD:  IMBALANCE!!!!!!
-------------------------------------------------------------------------------------------------------------------------------
              
              L1 = BCE()
              L2 = MSE()

              L = w1*L1 + w2*L2  <-----------what is w1 and w2??? The weigths they spoke about...
                
              L.backward()

Weights:  I was not familiar with the impact of dynamically adjusting weights to handle class imbalance.
Weights:  Hint: Similar to HW3 the positive and negative samples are highly imbalanced. Try using the FOCAL LOSS.
                
Weights:  alter the weight of the loss function to account for the class imbalance and not the weights of the network
More:  https://piazza.com/class/ksjhagmd59d6sg?cid=618
                
                
Hint: Use the sigmoid and BCEWithLogitsLoss
NEW HINT:  sigmoid function would be helpful when implementing the FOCAL LOSS, if you won’t use focal loss, you can just use BCEWithLogitsLoss.
Whatever:  Focal loss or BCEWithLogitsLoss should be used while we are training the model, not in the implementation of the model itself
On Average Precision (AP): AP is not a requirement for model validation, just train the Detector() as in HW3, and you can validate 
training with the loss, or by visualizing the detection by using the log function provided in train.py.
Source:  https://piazza.com/class/ksjhagmd59d6sg?cid=663

FOCAL LOSS:  IMBALANCE - MOST PIXELS ARE BACKGROUND (or TRACK, etc)!!!!!!!!!!!!  IMBALANCE HERE!!
        
FOCAL LOSS: WHAT IS IT?  (FOR CLASS IMBALANCE) https://amaarora.github.io/2020/06/29/FocalLoss.html
FOCAL LOSS: WHAT IS IT?  (FOR CLASS IMBALANCE) https://amaarora.github.io/2020/06/29/FocalLoss.html
FOCAL LOSS: WHAT IS IT?  (FOR CLASS IMBALANCE) https://amaarora.github.io/2020/06/29/FocalLoss.html
                
 *Whereasm, what Focal Loss does is that it makes it easier for the model to predict things without being 80-100% sure that this object is “something”. 
 *In simple words, giving the model a bit more freedom to take some risk when making predictions. This is particularly important when dealing with highly IMBALANCED
 *datasets because in some cases (such as cancer detection), we really need to model to take a risk and predict something even if the prediction turns out to be a
 *False Positive.  

 *Therefore, Focal Loss is particularly useful in cases where there is a class imbalance. Another example, is in the case of *OBJECT DETECTION* when most pixels are usually
 *background and only very few pixels inside an image sometimes have the object of interest. 

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

FOCAL LOSS:  You will need to implement it yourself, but there are plenty of resources with respect to implementing it in conjunction with PyTorch with a simple Google search.
FOCAL LOSS:  https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327  
FOCAL LOSS: Remember the alpha to address class imbalance and keep in mind that this will only work for binary classification.
                
                         CLASS torch.nn.BCEWithLogitsLoss(weight=None,
                                                          size_average=None, 
                                                          reduce=None, 
                                                          reduction='mean', 
                                                          pos_weight=None)

This loss combines a Sigmoid layer and the BCELoss in one single class.
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, 
we take advantage of the log-sum-exp trick for numerical stability.
       
loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(input, target)
output.backward()

-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------

  Less than 150 Epochs

  
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------

Detection Test
Is there a way we can change the detection test in the test.py to in such a way that it will always get full points unless we have our
detect or peak extraction wrong? Finding it impossible to debug what is wrong and thought passing the correct label to the test would help in such a case.

----->I’m not sure if there’s an easy way to do this, but I’m also not sure I understand why you want to. The extract_peak function is already graded separately.
----->For the detect function you could consider creating your own test case (creating an image with red/green/blue dots and using that insted of your model output)
----->and seeing if the function returns the expected output.

  
-------------------------------------------------------------------------------------------------------------------------------
                                                HEATPMAPS
-------------------------------------------------------------------------------------------------------------------------------
kel76y
DEFINING HEATMAPS

1. Predict a dense HEATMAP of object centers, 
2. Each “PEAK” (local maxima) in this HEATMAP corresponds to a detected object.
3. The input to extract_peak is a 2d HEATMAP, the output is a list of PEAKS with SCORE, x and y location of the center of the PEAK.  (But per #5, HEATMAP is NOT a tensor?)
4. The SCORE is the value of the HEATMAP at the peak, and is used to rank detections later. A peak with a high score is a more important detection.
5. Hint: Max pooling expects an order-4 tensor as an input, use heatmap[None,None] to convert the input HEATMAP to an order-4 tensor.
6. You model should take an image as an input and predict a HEATMAP for *****each object.***
7. Each heatmap is an independent 0-1 output map
    TRAINING: 
8. You can convert the *DETECTION ANNOTATIONS* to a heatmap by calling the dense_transforms.ToHeatmap transform.
9. ToHeatmap changes just the label and leaves the image unchanged. Call ToTensor to change the image. 
10.class ToHeatmap(object): def __init__(self, radius=2): self.radius = radius   (dense_transforms.py)
11.The value of the heatmap at the local maxima (peak score) is a good confidence measure. Use the extract_peak function to find detected objects.
12.ANNOTATIONS:  Unlike HW3, the annotations for this homework are not yet in a format that can be batched by pytorch. 
13.ANNOTATIONS:  dense_transforms.py also contains several other annotations that you might find useful.

OTHER DEFINITIONS OF HEATMAP:

https://arxiv.org/pdf/2101.03541.pdf

1.     The task now was to use the location information of the previously described .txt file to create so-called heatmaps which represent the ideal output
we want from our neural network for each picture.

2.   By taking the coordinates of the .txt files and putting a Gaussian
distribution around them, we could create the desired heatmaps.

3.   The purpose of these heatmaps is to help us tell the CNN how good or bad it is currently doing at
telling us the location of the ball on the labyrinth, by providing the perfect solution.

4. The heatmaps we are using to represent our labels and network output are, again, just a 2D-matrix of numbers,
in which every number stands for the activation of a pixel

5.  The whole heatmap can be looked at as a sort of ”probability distribution” on
where the CNN thinks the ball is.


PER THE PROFESSOR IN OBJECT TRACKING:

1. We simply feed the input image to a fully convolutional network [37, 40] that generates
a heatmap. Peaks in this heatmap correspond to object centers.

2.  We simply extract local peaks in the keypoint heatmap

3.  Our aim is to produce a keypoint heatmap Yˆ ∈ [0, 1] W/R × H/R  × C, where R is the output stride and C is the
number of keypoint types.

4.  Keypoint types include C = 17 human joints in human pose estimation, or ****** C = 80 object categories in object detection.

5. We train the keypoint prediction.... For each ground truth keypoint p ∈ R2 of class c, we compute a low-resolution equivalent p˜ = p/R (lower bound?).  

6.. We then splat all ground truth keypoints onto a HEATMAP. 

7. At inference time, we first extract the peaks in the heatmap for EACH category independently. 
                                                                                                                                      
                                                                                                                                      
USEFUL POSTS:
1. https://piazza.com/class/ksjhagmd59d6sg?cid=655   "Detection - About the assignment specs"   
   "So is the raw output of our segmentation FCN called the predicted heatmap? In HW3 we took this output"
   "and just did a softmax and cross entropy loss on it to find which pixel belonged to which class." 
    "So are we taking this same output but this time, finding the peaks?"
                                                                                                                                                          

2. https://piazza.com/class/ksjhagmd59d6sg?cid=634
"

peak extract

The Heatmap has 3 channels. MaxPool2d operates channel wise and will return 3 channels..  once we compare the maxpool with the heatmap,
should  we use a heuristic to combine the channels (eg summation), do we need a separate convolutional layer to deep learn the right way to 
combine channel information, just pass three separate channels, or something else? 

I am not sure why you say the heatmap has only two dimensions.   Using batchsize of 100, this is what comes out of the dataloader.

100,3,96,128
100,3,96,128
100,3,96,128

Can you please resolve my question, how to apply peak-extract to a multi-channel heatmap.???
---------->If you're just working on the extract_peak function, we should be passing one channel at a time to the extract_peak function.
---------->So the "heatmap" input to extract_peak function is just HxW
---------->As described in the starter code, extract_peak() is a function that works for one channel of heatmap (H x W).
---------->In detect(), you will need to let an image (3 x H x W) pass the called forward() and get a 3-channel heatmap (3 x H x W).
---------->Then in detect() you will apply the extract_peak to the 3-channel heatmap three times (by a for loop or list them), then return 3 list of detections.

Q. is each channel of the heatmap a color (RGB) or does it a correspond to a class of objects?
------------>each channel corresponds to a class.


"                                                                                                                                      

                                                                                                                                      
HOW TO CODE THIS:
                                                                                                                                      
1.    Implement __init__ and forward
2.    detect and detect_with_size later.

                                                                                                                                     
                                                                                                                                      
                                                                                                                                     
                                                                                                                                      
            
                                                                                                                                      
                                          THE CODE  (MODELS.Py) ---- NOTE:  NO CNN????
                                                                                                                                      
def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    
       """
       THIS IS USED IN CLASS DETECTOR, DETECT METHOD
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap) - THIS IS NOT A TENSOR? ********
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
       heatmap value at the peak. Return no more than max_det peaks per image
       
       extract_peak-->  PARAMETER @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
        if it said only return points equal to the   maxpool,  it would make sense. But how can a point exceed the maxpool centered at that point
       
       -------->The hint means you should only return a value if it is larger than all values in a max_pool_ks x max_pool_ks  square around it.
       -------->E.g. if we had a max_pool_ks of 3:

1 3 2
5 4 1
2 1 3

the 4 in the center should not be detected as a peak since the 5 to the left of it is larger.
       BUT--------------------->5 is a global maximum but it would flunk that test since 5 is not greater than the maxpool of 5
                 REPLY: I see what you mean. The value does not have to be larger than itself, it just has to be the maximum value in that window.
                        If we were looking at an example like

                                             1 3 2
                                             4 5 1
                                             2 1 3

                       with a window size of 3, 5 would be a valid peak.
       
    """
    raise NotImplementedError('extract_peak')

                                                                                                                                                                                                                          
 ---------------    *************DETECTOR CLASS*********  -----------------------------------
                                                  
"""

You model should take an image as an input and predict a heatmap for ****each object.*** (e.g. six objects, six heatmaps). 
Each heatmap is an independent 0-1 output map. The heatmap is zero almost everywhere, except at object centers. 
Use a model similar to your solution to HW3 or the master solution
                                                                                                                                      
"""                                                                                                                                      
                                                                                                                                      
class Detector(torch.nn.Module):
                                                                                                                                      
                                                                                                                                      
    def __init__(self):     #DO THIS FIRST!!!!!!!!!!!!
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        raise NotImplementedError('Detector.__init__')

---------------    DETECTOR CLASS, FORWARD() METHOD  -----------------------------------

    def forward(self, x):  #DO THIS FIRST!!!!!!!!!!!!
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        raise NotImplementedError('Detector.forward')

                                                                                                                                      
 ---------------    DETECTOR CLASS, DETECT() METHOD  -----------------------------------
                                                                                                                                      
    def detect(self, image):   #------------->Use extract_peak here, ONE SINGLE IMAGE
                                                                                                                                      
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
                
           def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
                returns List of peaks [(score, cx, cy), ...], 
            
---------->As described in the starter code, extract_peak() is a function that works for one channel of heatmap (H x W).
---------->In detect(), you will need to let an image (3 x H x W) pass the called forward() and get a 3-channel heatmap (3 x H x W).
---------->Then in detect() you will apply the extract_peak to the 3-channel heatmap three times (by a for loop or list them), then return 3 list of detections. 


        """
        raise NotImplementedError('Detector.detect')

                                                                                                                                      
      def detect_with_size   #-----------------*EXTRA CREDIT
                                                                                                                                      
  """
  You can earn some extra credit by implementing a full-fledged object detector. To do so you’ll need to predict the size of the object in question.
  The easiest way to do this is to predict two more output channels, in addition to the 3-channel heatmap. We will evaluate AP at an intersection over union (overlap) of 0.5.
  
  """
