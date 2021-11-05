import torch
import torch.nn.functional as F


#max_det = 100???
#max_det = 100???
#max_det = 100???

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=30):

    print("----------------------EXTRACT PEAK CALLED------------------------")

    detection_list = []
    k=max_det

    Maxpool =  torch.nn.MaxPool2d(kernel_size=max_pool_ks, return_indices=True, padding=max_pool_ks//2, stride=1)
    
    # Original HEATMAP SHAPE IS    torch.Size([96, 128])
    heatmap_temp = heatmap[None, None] 
    #for Maxpool, shape is NOW torch.Size([1, 1, 96, 128]
    heatmap_temp = heatmap_temp.to(heatmap.device)

    maxpooled_heatmap, maxpool_indices =  Maxpool(heatmap_temp)   #torch.Size([1, 1, 96, 128])
   
    maxpooled_heatmap = maxpooled_heatmap.to(heatmap.device)
   
    sorted_scores, sorted_idx = maxpooled_heatmap.view(-1).sort()

    """
    1. iterate through maxpooled_heatmap and heatmap
    2. if maxpooled_heatmap[i][j] == heatmap[i][j] then these are the coordinates of a peak
         i.  if the value at this location is greater than -5,
                    a. store coordinates and the value, append a list I suppose 
    """  




    #find the index, call it idx_min, of the first score which is > min_score
    
    idx_min = 0
    for i in range(len(sorted_scores)):
      if sorted_scores[i] > min_score:
        idx_min = i
        print (f'{i} ...I found the first score > min_score:  {sorted_scores[i]}, its index is {sorted_idx[i]}. idx_min is {i}')
        break

    #feed sorted_scores[idx_min:] into top scores
    topk, indicesTOP = torch.topk(sorted_scores[idx_min:], k)

    #find these topK (30) values in the original heatmap

    print (f'>>>>>>>>>>>>>>>>>>>   The top 30 scores above -5 are {topk}')
    print (f'This is the sorted_scores list, {sorted_scores}')



    #print(f'This is the list of sorted maxpool scores, length {len(sorted_scores)} followed by indices --->{sorted_scores}INDICES------>INDICES:{sorted_idx}') 
    
    #Nov 5, 2021
    #1 Iterate through max_pool, omit all scores < min_score
      #heatmap.view(-1), sorted?
      #sorted, sorted_idx = maxpooled_heatmap.view(-1).sort()


    #2 Find Topk, k=max_det  
    
    #3 Find cx, cy in heatmap for those  topk scores

    #3done

    #topk, indicesTOP = torch.topk(maxpooled_heatmap.view(-1), k)

    #detection_list = [(3,4,5, 0, 0), (3,4,5, 0, 0), (3,4,5, 0, 0) ]
    #detection_list.append((3, 3, 3, 0, 0)) 

    return(detection_list)


#END----------------------END  extract_peak

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

            #print (f'                              -----:  IN FORWARD OF CNN BLOCK, X shape IS  {x.shape}  ')
            #print (f'--------------------------------------------------------------------------------------')

            output = self.c1(x)    #NOV 1, 2021:  NEVER REACHES THIS POINT!
            #print (f'1   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = self.b1(output)
            #print (f'2   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = F.relu(output)
            #print (f'3   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = self.c2(output)
            #print (f'4   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = self.b2(output)
            #print (f'5   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = F.relu(output)
            #print (f'6   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = self.c3(output)
            #print (f'7   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            #output = self.b3(output) + self.skip(x)
            #print (f'8   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = F.relu(output)
            #print (f'9   DOWNCONV CNN BLOCK-----:  The OUTPUT mean IS  {output.mean()}  ')
            output = output.to(x.device)
            #print (f'              DOWNCONV CNN BLOCK, RETURNING:   OUTPUT of shape  {output.shape}  ')
    
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
                #print(f'                         ADDING DECONV SKIP, c is {c}<<<<<<<<<<<<<<<<<<<<')
                c += skip_layer_size[i]
                #print(f'                         JUST ADDED DECONV SKIP, c is NOW {c}<<<<<<<<<<<<<<<<<<<<')

      
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)      #MAY NEED TO CHANGE THIS OUTPUT NOV 1, 2021
#------------------------------------------------------------------------------>END CONSTRUCTOR FOR DETECTOR NETWORK
       





#------------------------------------------------------------------------------>BEGIN  Detector.FORWARD() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USES Detector.DETECT() METHOD FOR DETECTION
#OCT 30, 2021

    def forward(self, img):
        
      #print (f'--------------------------------------------------------------------------------------')
      #print("||||||||||||STEP 3a||||||||||   INSIDE DETECTOR()--->FORWARD(), will call CNN BLOCK NEXT, DECONV")
      #print (f'--------------------------------------------------------------------------------------')
      #print (f'                ------>>>>>>>>>>>, image shape is {img.shape}, will be normalized, THEN passed to CNN BLOCK and then DECONV')
        
      #print (self.detect(img))   #CALLING DETECT() HERE FOR TEST PURPOSES           

      heatmap = (img - self.input_mean[None, :, None, None].to(img.device)) / self.input_std[None, :, None, None].to(img.device)
      
      up_activation = []
      
    
      for i in range(self.n_conv):             #in range 4 basically.
        
        # Add all the information required for skip connections
        up_activation.append(heatmap)
        heatmap = self._modules['conv%d'%i](heatmap)   
        
      #print (f'           FORWARD()--->  AFTER DOWNCONV CNN BLOCK, HEATMAP MEAN IS -----:  {heatmap.mean()}  ')
      

      for i in reversed(range(self.n_conv)):
      
        heatmap = self._modules['upconv%d'%i](heatmap)
        # Fix the padding
        heatmap = heatmap[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
        # Add the skip connection
        if self.use_skip:
          heatmap = torch.cat([heatmap, up_activation[i]], dim=1)
      
      output = self.classifier(heatmap)   #returns heatmap ([32, 3, 96, 128])

      #print("||||||||||||STEP 3b||||||||||   INSIDE DETECTOR()--->FORWARD(), Just created a heatmap with Image USING CONV/DECONV Layers")
      #print (f'--------------------------------------------------------------------------------------')
      
      return output

#ABOVE IS DETECTOR->FORWARD RETURN VALUE heatmap ([32, 3, 96, 128])


#------------------------------------------------------------------------------>BEGIN  Detector.DETECT() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USED BY Detector.FORWARD() METHOD ABOVE FOR DETECTION

    def detect(self, image):

        List_of_detection_lists =[]

        
        #print (f'--------------------------------------------------------------------------------------')
        #print("||||||||||||STEP 2||||||||||   INSIDE DETECTOR()--->DETECT()")
        #print (f'--------------------------------------------------------------------------------------')
        #print (f'                ------>>>>>>>>>>>, image shape is {image.shape}, will be passed to FORWARD() Next ')
        
        three_channel_heatmap = self.forward(image)


        #print("||||||||||||STEP 4||||||||||   JUST CALLED DETECTOR()--->FORWARD(), BACK IN DETECT()")
        #print (f'--------------------------------------------------------------------------------------')
        #print (f'                ------>>>>>>>>>>>, 3 channel heatmap shape is {three_channel_heatmap.shape}')
        #print (f'                ------>>>>>>>>>>>, NOW WILL GO THROUGH ALL 3 CHANNELS AND CALL EXTRACT_PEAKS()')
        
      
        #heatmap shape is torch.Size([1, 3, 96, 128])

        for i in range (3):         
          List_of_detection_lists.append(extract_peak(three_channel_heatmap[0][i]))

        #print('                                  ^^^^^^This is the list of lists, -> {List_of_detection_lists}')
        #print(List_of_detection_lists)
        
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
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    #print("||||||||||||STEP 1||||||||||   IN MAIN(), GOING TO LOAD DATA and ITERATE, AND CALL DETECTOR() ")
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):    #called about 11 Times
        
        im, kart, bomb, pickup = dataset[i]
        #print(f'              MEAN: --->In MAIN(), the mean of the Image just pulled is {im.mean()} ')
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
    
    show()
