import torch
import torch.nn.functional as F


#max_det = 100???
#max_det = 100???
#max_det = 100???

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=30):

    """
    x2 = torch.tensor([2,2,77,89,1,7,65,100,12,500])
    f, indices = torch.topk(x2,3)
    """
    
    #IMPORTANT
    #@max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks 
    #window around the point
    #https://piazza.com/class/ksjhagmd59d6sg?cid=620
    #The hint means you should only return a value if it is larger than 
    #all values in a max_pool_ks x max_pool_ks  square around it.

    #switched it to functional max_pool2d????
    Maxpool =  torch.nn.MaxPool2d(kernel_size=max_pool_ks, return_indices=True, padding=max_pool_ks//2, stride=1)
    
    heatmap2 = heatmap[None, None] #for Maxpool, shape is torch.Size([1, 1, 96, 128]
    maxpooled_heatmap, indices =  Maxpool(heatmap2)
    
    k=max_det
    """
      maxpool2d() helps you identify the local maximum by *distributing* the max value to all
      indexes in a given window. In the provided example, we want to know that 3 is indeed not a 
      local maxima. When maxpool2d() returns the modified tensor with 5 distributed over the 
      index where 3 was originally located, it becomes clear that 3 is not greater than or equal
      to its neighbors.  
      
      You need to make use of the information maxpool2d() provides you with, and 
      ***-->generate a new tensor suitable for ***TOPK()***. 
      You should not pass the raw output of maxpool to topk. 
      
      The slides also illustrates how you can find the **true** indexes to
      generate  the new tensor
    
    """

    topk, indicesTOP = torch.topk(maxpooled_heatmap.view(-1), k)

    print("11111111111111111111222222222222222233333333333333555555555555555555555555555")
    print(f'               HEATMAP SHAPE IS    {heatmap.shape}')
    print(f'               MAXPOOLED HEATMAP SHAPE IS    {maxpooled_heatmap.shape}')
    print(f'               MAXPOOLED HEATMAP MEAN  IS    {maxpooled_heatmap.mean()}')
    print(f'               Topk is    {topk}')
    print(f'               Topk indices is    {indicesTOP}')
    print(f'               MAXPOOL heatmap indices SHAPE is    {indices.shape}')
    print(f'               MAXPOOL heatmap indices is    {indices}')
    
    
    #print(f'       This is the MAXPOOLED HEATMAP................{maxpooled_heatmap}')
    #flatten
    #topk

    detection_list = [(3,4,5, 0, 0), (3,4,5, 0, 0), (3,4,5, 0, 0) ]

    detection_list.append((3, 3, 3, 0, 0)) 

    
    print ("You have succesfully called extract_peak as follows:  Detector-->Forward()--->Detect()--->extrac_peak")
    print(f'                          SHAPE OF HEATMAP in EXTRACTPEAK is {heatmap.shape}, of HEATMAP2, {heatmap2.shape}')
    print (f'Returning this list:, {detection_list}')

    return(detection_list)



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

            print (f'NOV 2, 2021-----:  IN FORWARD OF CNN BLOCK, X shape IS  {x.shape}  ')
            

            output = self.c1(x)    #NOV 1, 2021:  NEVER REACHES THIS POINT!
            #print (f'1   NOV 2, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = self.b1(output)
            #print (f'2   NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = F.relu(output)
            #print (f'3    NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = self.c2(output)
            #print (f'4   NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = self.b2(output)
            #print (f'5    NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = F.relu(output)
            #print (f'6    NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = self.c3(output)
            #print (f'7    NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = self.b3(output) + self.skip(x)
            #print (f'8  NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = F.relu(output)
            #print (f'FINAL BEFORE RETURN    NOV 1, 2021-----:  The OUTPUT shape IS  {output.shape}  ')
            output = output.to(x.device)
            print (f'              DOWNCONV CNN BLOCK, RETURNING:   OUTPUT of shape  {output.shape}  ')
            
            
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

    class UpBlock(torch.nn.Module):          #DECONVOLUTION

        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))
            
#-------------->DETECTOR  UPBLOCK CLASS END 


#NOV 2, 2021, CHANGED n_output_channels=5 to 3!!!!!

#------------------------------------------------------------------------------>BEGIN CONSTRUCTOR FOR DETECTOR NETWORK
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip             #use_skip initially TRUE
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]  #layers[:-1] is [16, 32, 64], so skip_layer_size is [3, 16, 32, 64]
        
        #CONVOLUTIONAL LAYERS - CNN BLOCKS
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))   #ONLY USING NON-LINEAR CNN BLOCKS!
            c = l
        
        #DECONVOLUTIONAL LAYERS
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]

      
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)      #MAY NEED TO CHANGE THIS OUTPUT NOV 1, 2021
#------------------------------------------------------------------------------>END CONSTRUCTOR FOR DETECTOR NETWORK
       





#------------------------------------------------------------------------------>BEGIN  Detector.FORWARD() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USES Detector.DETECT() METHOD FOR DETECTION
#OCT 30, 2021

    def forward(self, img):
        

      print("MODELS.PY Making a Prediction/Detection NOW, INSIDE FORWARD")
      #print (self.detect(img))   #CALLING DETECT() HERE FOR TEST PURPOSES           

      heatmap = (img - self.input_mean[None, :, None, None].to(img.device)) / self.input_std[None, :, None, None].to(img.device)
     
      
      up_activation = []
      

      print ("------------------DETECTOR-->FORWARD() NOW (MODELS.PY)---------------------------------------------")
    
      
      for i in range(self.n_conv):             #in range 4 basically.
        
        #print(f'            In detector->Forward FIRST LOOP Number {i+1}')
        # Add all the information required for skip connections
        up_activation.append(heatmap)
       # print(f'            In detector->Forward LOOP Number {i+1} AFTER append()')
        heatmap = self._modules['conv%d'%i](heatmap)   
        

        #print(f'              In detector->Forward LOOP Number {i} AFTER heatmap input went through DownConv CNN Block')

      for i in reversed(range(self.n_conv)):
        #print(f'                       In REVERSED detector->Forward SECOND LOOP, UPCONV, Number {i}')
        heatmap = self._modules['upconv%d'%i](heatmap)
        # Fix the padding
        heatmap = heatmap[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
        # Add the skip connection
        if self.use_skip:
          heatmap = torch.cat([heatmap, up_activation[i]], dim=1)

      return self.classifier(heatmap)   #returns heatmap ([32, 3, 96, 128])

#ABOVE IS DETECTOR->FORWARD RETURN VALUE heatmap ([32, 3, 96, 128])





#------------------------------------------------------------------------------>BEGIN  Detector.DETECT() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USED BY Detector.FORWARD() METHOD ABOVE FOR DETECTION

    def detect(self, image):

        three_channel_heatmap = self.forward(image)
        
        List_of_detection_lists =[]

        print (f'             NOV 3, 2021------>>>>>>>>>>>, heatmap shape is {three_channel_heatmap.shape} ')
        #heatmap shape is torch.Size([1, 3, 96, 128])

        for i in range (3):
          #print(f'                        ***Calling extract peak at iteration {i+1}')
          List_of_detection_lists.append(extract_peak(three_channel_heatmap[0][i]))

        print('                                  ^^^^^^This is the list of lists, -> {List_of_detection_lists}')
        print(List_of_detection_lists)
        
        return (List_of_detection_lists)              
   



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

#---------------------------------------------------MAIN----------------------------------------------------

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
        print(f'                          IN THE MAIN() LOOP NOW, NUMBER {i}')  #called about 11 times
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
        detections = model.detect(im.to(device))  #****************************************************************<<<<<<
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    print("About to Call SHOW() DID you see anything?")
    show()
