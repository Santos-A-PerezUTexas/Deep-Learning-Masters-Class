import torch
import torch.nn.functional as F


#max_det = 100??
#max_det = 100??
#max_det = 100??

def get_idx(idx, shape):
    
    
    res = []
    N = shape[0]*shape[1]
    for n in shape:
        N //= n
        res.append(idx // N)
        idx %= N
    return tuple(res)

def extract_peak(heatmap, max_pool_ks=3, min_score=-5, max_det=10000):

    print ("----------------------EXTRACT PEAK CALLED------------------------")

    detection_list = []
    k=max_det
    Maxpool =  torch.nn.MaxPool2d(kernel_size=max_pool_ks, return_indices=True, padding=max_pool_ks//2, stride=1)
    
    #print(f'k is {k}, max_pool_ks is {max_pool_ks}, and max_det is {max_det}')
    #print(f'heatmap shape {heatmap.shape}')
    heatmap4D = heatmap[None, None]   # reduced=torch.Size([1, 1, 96, 128], heatmap=torch.Size([96, 128])
    heatmap4D = heatmap4D.to(heatmap.device)
    maxpooled_heatmap, maxpool_indices =  Maxpool(heatmap4D)  

    maxpooled_heatmap = maxpooled_heatmap[0][0]   #torch.Size([96, 128])
    maxpooled_heatmap = maxpooled_heatmap.to(heatmap.device)

    peak_tensor = torch.where(heatmap==maxpooled_heatmap, 1, 0)     #0-1 Heatmap W/ Peaks
    peak_tensor = peak_tensor.to(heatmap.device)
    z = torch.zeros(heatmap.shape).to(heatmap.device)
    z.fill_(-1000000)
   
    peaks = torch.where((peak_tensor)==1, heatmap, z).to(heatmap.device)
    #peaks = torch.where(peaks>min_score, heatmap, z)

    #print(f'sorted peaks top 5 is {sorted(peaks.view(-1))[-5:]}')
    #print(f'peaks top 5 with topk {torch.topk(peaks.view(-1), 5)[0]}')
    
    #topk = torch.topk(peaks.view(-1), k)[0]
    cx, cy = get_idx(torch.topk(peaks.view(-1), k)[1], heatmap.shape)
     
    for i in range(k):
      #print(heatmap[cx[i]][cy[i]])
      if peaks[cx[i]][cy[i]] > min_score:
        detection_list.append((peaks[cx[i]][cy[i]], cx[i], cy[i],0 ,0))
      

    #print(f'--------cx is {cx}---------cy is {cy}<-----------------------')
    #print ("CALLING FOR LOOP------")
    #print (f'heatmap shape size is {heatmap.shape[0]*heatmap.shape[1]} ')
    #for i in range (heatmap.shape[0]*heatmap.shape[1]):
     # if peak_tensor.view(-1)[i]:
      #    cx, cy=get_idx(i, heatmap.shape)
       #   if heatmap[cx][cy] > min_score:
        #    detection_list.append((heatmap[cx][cy], cx, cy,0 ,0))
    
    print ("DETECTION LIST LENGTH AND CONTENTS")
    print (len(detection_list))  #should be < max_det
    print (detection_list)
    #print ("SORTED DETECTION LIST, first three entires")
    #print(sorted(detection_list)[-3:])

    if k < len(detection_list): 
      print(".....................TRUNCATING LONG LIST...........................")
      print(".....................TRUNCATING LONG LIST...........................")  
      print(".....................TRUNCATING LONG LIST...........................")  
      print(".....................TRUNCATING LONG LIST...........................")                      
      detection_list = sorted(detection_list)[-k:]

    
    #print ("EXTRACT_PEAK()---------------DETECTION LIST LENGTH")
    #print (len(detection_list))  #should be < max_det
    #print (detection_list)
    #print ("SORTED DETECTION LIST LENGTH AND CONTENTS")


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
          return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))
           
           
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
        
     
      heatmap = (img - self.input_mean[None, :, None, None].to(img.device)) / self.input_std[None, :, None, None].to(img.device)
      
      up_activation = []
      
    
      for i in range(self.n_conv):             #in range 4 basically.
        
        # Add all the information required for skip connections
        up_activation.append(heatmap)
        heatmap = self._modules['conv%d'%i](heatmap)   
           

      for i in reversed(range(self.n_conv)):
      
        heatmap = self._modules['upconv%d'%i](heatmap)
        # Fix the padding
        heatmap = heatmap[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
        # Add the skip connection
        if self.use_skip:
          heatmap = torch.cat([heatmap, up_activation[i]], dim=1)
      
          
      return self.classifier(heatmap)  #returns heatmap ([32, 3, 96, 128])

#ABOVE IS DETECTOR->FORWARD RETURN VALUE heatmap ([32, 3, 96, 128])


#------------------------------------------------------------------------------>BEGIN  Detector.DETECT() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USED BY Detector.FORWARD() METHOD ABOVE FOR DETECTION

    def detect(self, image):

        List_of_detection_lists =[]

        
        three_channel_heatmap = self.forward(image)

            
        #heatmap shape is torch.Size([1, 3, 96, 128])

        for i in range (three_channel_heatmap.shape[1]):         
          List_of_detection_lists.append(extract_peak(three_channel_heatmap[0][i]))
    
        
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
