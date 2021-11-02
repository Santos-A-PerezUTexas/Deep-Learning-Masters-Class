import torch
import torch.nn.functional as F

#Oct 22 2021

#torch.nn.functional.max_pool2d(*args, **kwargs)
#Applies a 2D max pooling over an input signal composed of several input planes.
#When using the 2D Global Max Pooling block, all of the input channels are combined 
#into one large pool (i.e., the Horizontal pooling factor x the Vertical pooling factor), 
#and a single maximum value is computed for each channel.
#https://www.quora.com/What-is-Max-Pooling-2D

#torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
#Returns the k largest elements of the given input tensor along a given dimension.



def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image

      Use *maxpool2d()* and *topk()*:

      The Heatmap has 3 channels. MaxPool2d operates channel wise and will return 3 channels..  once we *compare* the 
      maxpool with the heatmap, should  we use a heuristic to combine the channels (eg summation), do we need a 
      separate convolutional layer to deep learn the right way to  combine channel information, just pass three 
      separate channels, or something else?

      how to apply peak-extract to a multi-channel heatmap
          ---------->If you're just working on the extract_peak function, we should be passing one channel at a time to the extract_peak function.
          ---------->So the "heatmap" input to extract_peak function is just HxW
          ---------->As described in the starter code, extract_peak() is a function that works for one channel of heatmap (H x W).
          ---------->In detect(), you will need to let an image (3 x H x W) pass the called forward() and get a 3-channel heatmap (3 x H x W).
          ---------->Then in detect() you will apply the extract_peak to the 3-channel heatmap three times (by a for loop or list them), then return 3 list of detections.


      As noted in the slides, maxpool2d() helps you identify the local maximum by *distributing* the max value to all
      indexes in a given window. In the provided example, we want to know that 3 is indeed not a local maxima. When 
      maxpool2d() returns the modified tensor with 5 distributed over the index where 3 was originally located, it 
      becomes clear that 3 is not greater than or equal to its neighbors.  You need to make use of the information 
      maxpool2d() provides you with, and generate a new tensor suitable for ***TOPK()***. 
      You should not pass the raw output of maxpool to topk. The slides also illustrates how you can find the true indexes to
      generate  the new tensor.

      After max pooling(), how does one pick out the actual maxima from the returned tensor? The only way I can think of is
      by basically manually doing what maxpool2d() is doing, but that obviously defeats the purpose. I know the next steps are
       to pass a ***flattened** tensor to *topk()*, but am missing something in the order of operations here.
               So you have two tensors after max pooling, the pool and the original heat map. There are very simple
               operations and checks you can do you can *compare* the two to isolate the peaks. I did this comparison before 
               passing it into *topk()*


      **********torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
                  Returns the k largest elements of the given input tensor along a given dimension.
      **********torch.nn.MaxPool2d(kernel_size=max_pool_ks=7, stride=None, padding=0, dilation=1, return_indices=True, ceil_mode=False)
                        **return_indices – if True, will return the max indices along with the outputs. 
                        **Max pooling expects an order-4 tensor as an input, use heatmap[None,None] to convert the input heatmap to an order-4 tensor.
                        **Input size (N, C, H, W), output (N, C, H_{out}, W_{out}) 

             1. Heatmap [None, None] to convert the input heatmap to an order-4 tensor.
             2. Maxpool = Maxpool2d (kernel_size=max_pool_ks=7, return_indices=True) #return indices?
             3. maxpooled_heatmap =  MaxPool(Heatmap) #identify local maximum by *distributing* max value to all indexes in a given window
             3. flattened_tensor = Flattenned maxpooled_heatmap
             4. topk(flattened_tensor, k, dim=None, largest=True, sorted=True, *, out=None)
             5.  What is k for topk()?
                      k must <= max_det
                      k must <=the number of points output by max_pool2d. 
                      https://piazza.com/class/ksjhagmd59d6sg?cid=256

             6. Where do I use these min_score=-5, max_det=100 ????
             7. Return no more than max_det peaks per image
             8. 
             9. Return:  (score, cx, cy) is a tuple with three numbers

             RETURN (List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the heatmap value at the peak. (score, cx, cy) is a tuple with three numbers)
                        Return no more than max_det peaks per image

           EXAMPLE:

              1.  x = torch.tensor([[0,0,0,0,0], [0,1,2,3,0], [0,4,5,6,0],[0,7,8,9,0],[0,0,0,0,0]])
              2. Maxpool = torch.nn.MaxPool2d (kernel_size=3)
              3. x.vew(-1) is the flattened x, but not what we want, we want x[None,None]
              4. So y = Maxpool(x[None,None]) ---> extract_peaks() takes in a 2-D tensor. Expand the 2-D tensor to a 4-D tensor by using 4d_tensor = 2d_tensor[None, None].
              5. ***RuntimeError: "max_pool2d" not implemented for 'Long'  (locally when tested)
    
    """ 
    return("You have succesfully called extract_peak as follows:  Detector-->Forward()--->Detect()--->extrac_peak")



        #--------------------------------------BEGIN CNN

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            print (f'NOV 1, 2021-----:  The DEVICE of X IS  {x.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of X IS  {x.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of X IS  {x.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of X IS  {x.device}  ')

            output = self.c1(x)
            output = self.b1(output)
            output = F.relu(output)
            output = self.c2(output)
            output = self.b2(output)
            output = F.relu(output)
            output = self.c3(output)
            output = self.b3(output) + self.skip(x)

            output = F.relu(output)
            
            print (f'NOV 1, 2021-----:  The DEVICE of OUTPUT IS  {output.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of OUTPUT IS  {output.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of OUTPUT IS  {output.device}  ')
            output = output.to(x.device)
            print (f'NOV 1, 2021-----:  The DEVICE of OUTPUT IS NOW  {output.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of OUTPUT IS  NOW {output.device}  ')
            print (f'NOV 1, 2021-----:  The DEVICE of OUTPUT IS  NOW {output.device}  ')
            
            return output
            #NOV 1, 2021: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same


  #CNN Classifier begins here, the FCN uses ONLY above block.  Nov, 1, 2021

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3):
        super().__init__()
        self.input_mean = torch.Tensor([0.3235, 0.3310, 0.3445])
        self.input_std = torch.Tensor([0.2533, 0.2224, 0.2483])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))   #The Network is composed of a series of blocks w/ three layers each and skip C
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)    #LINEAR LAYER!!!!!!!!???   Nov 1, 2021

    def forward(self, x):
        z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        return self.classifier(z.mean(dim=[2, 3]))


        #--------------------------------------END CNN






class Detector(torch.nn.Module):


#-------------->DETECTOR() UPBLOCK BEGIN

    class UpBlock(torch.nn.Module):

        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))
            
#-------------->DETECTOR  UPBLOCK CLASS END 




#------------------------------------------------------------------------------>BEGIN CONSTRUCTOR FOR DETECTOR NETWORK
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=5, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip             #use_skip initially TRUE
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]  #layers[:-1] is [16, 32, 64], so skip_layer_size is [3, 16, 32, 64]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))   #ONLY USING NON-LINEAR CNN BLOCKS!
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)
#------------------------------------------------------------------------------>END CONSTRUCTOR FOR DETECTOR NETWORK
       





#------------------------------------------------------------------------------>BEGIN  Detector.FORWARD() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USES Detector.DETECT() METHOD FOR DETECTION
#OCT 30, 2021

    def forward(self, x):
        
      #Implement a forward pass through the network, use forward for training
      #and detect for detection  
      #essentially be the same as HW3 downsample and then upsample back the resolution of the image   
      #You can use only forward() in training, not detect(). 
      #You can call detect() optionally for inference/visualization
      # purposes, but it seems to be "necessary" only in the sense that the grader will evaluate and score it. 
      #Yeah so detect() is purely for inference. Training portion is very similar to HW3


      print("MODELS.PY Making a Prediction/Detection NOW, INSIDE FORWARD, OCT 30 2021")
      print (self.detect(x))   #CALLING DETECT() HERE FOR TEST PURPOSES OCT 30 2021          

      #CODE MAKES IT UP TO HERE, OCT 30, 2021 ---? then calls GetItem??? What?
      print ("LINE 134 MODELS.PY----------------------------------------------")
      z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
      up_activation = []
      

      print ("LINE 139 MODELS.PY---------------------------------------------")
      print ("LINE 139 MODELS.PY---------------------------------------------")
      print ("LINE 139 MODELS.PY---------------------------------------------")
      print ("LINE 139 MODELS.PY---------------------------------------------")
      
      for i in range(self.n_conv):             #in range 4 basically.
        print(f'In detector->Forward FIRST LOOP Number {i}')
        # Add all the information required for skip connections
        up_activation.append(z)
        print(f'In detector->Forward LOOP Number {i} AFTER append()')

        #AT THIS POINT, GET_ITEM IS CALLED, WHY?  AND PROGRAM CRASHES, OCT 30, 2021
        #AT THIS POINT, GET_ITEM IS CALLED, WHY?  AND PROGRAM CRASHES, OCT 30, 2021
        #AT THIS POINT, GET_ITEM IS CALLED, WHY?  AND PROGRAM CRASHES, OCT 30, 2021
        
        #NOV 1, 2021 remove comment pund sign below
        z = self._modules['conv%d'%i](z)    #NOV 1, 2021: Input type (torch.cuda.FloatTensor) and 
        #Nov 1, 2021: weight type (torch.FloatTensor) should be the same
        #Also line 113

        print(f'In detector->Forward LOOP Number {i} AFTER Z')

      for i in reversed(range(self.n_conv)):
        print(f'In REVERSED detector->Forward SECOND LOOP Number {i}')
        z = self._modules['upconv%d'%i](z)
        # Fix the padding
        z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
        # Add the skip connection
        if self.use_skip:
          z = torch.cat([z, up_activation[i]], dim=1)



        print (f'OCT 30, 2021 -------------------------------->This is Z: {z}')
        print (f'OCT 30, 2021 -------------------------------->This is Z: {z}')
        print (f'OCT 30, 2021 -------------------------------->This is Z: {z}')
        
      return self.classifier(z)






#------------------------------------------------------------------------------>BEGIN  Detector.DETECT() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USED BY Detector.FORWARD() METHOD ABOVE FOR DETECTION

    def detect(self, image):
        """
          Implement object detection here.
           
           image:           3 x H x W image
           return:          Three list of detections [(score, cx, cy, w/2, h/2), ...], one list per class (karts, bombs, pickup),
                            return no more than 30 detections per image per class. Predict width and height
                            for *extra credit*. If you do not predict an object size, return w=0, h=0.
           
           Hint:  Use extract_peak() 
           Hint:  Return THREE python lists of tuples of (float, int, int, float, float) and not a pytorch
                  scalar.
                  
              The final step of your detector extracts local maxima from each predicted heatmap. Each local maxima corresponds to a positive detection.
              The function detect() returns a tuple of detections as a list of five numbers per class (i.e., **tuple of three lists): The confidence of the 
              detection (float, higher means more confident, see evaluation), the x and y location of the object center (center of the predicted bounding box), 
              and the ***size** of the bounding box (width and height). Your detection function may return up to 100 detections per image, each detection comes with a confidence. 
              You’ll pay a higher price for getting a high confidence detection wrong. The value of the heatmap at the local maxima (peak score) is a good confidence measure.
              Use the extract_peak function to find detected objects.
            
        """
           
                  #ToTheatmaps converts ALL THREE OF THESE DETECTIONS  to peak and size tensors, which are the HEATMAPS????.
                
                  #extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
                  #Extract local maxima (peaks) in a 2d heatmap.
                  #heatmap:               H x W heatmap containing peaks (similar to your training heatmap)
                  #max_pool_ks:           Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
                  #min_score:             Only return peaks greater than min_score
                  #return:                 List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                  #heatmap value at the peak. Return no more than max_det peaks per image
        
            #Oct 31, 2021
            #You can call detect() optionally for inference/visualization
            #purposes, but it seems to be "necessary" only in the sense that the grader will evaluate and score it. 
            #Yeah so detect() is purely for inference. Training portion is very similar to HW3

        
        print(extract_peak(image))
        return ("MODELS.PY: THIS STRING WAS RETURNED FROM detect():  Class Detector.Forward()--->detect() ****************OCT 30, 2021")              
   



def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
