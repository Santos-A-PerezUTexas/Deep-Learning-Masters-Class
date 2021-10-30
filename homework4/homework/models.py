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
    """
    return("You have succesfully called extract_peak as follows:  Detector-->Forward()--->Detect()--->extrac_peak")



        #--------------------------------------BEGING CNN

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

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3):
        super().__init__()
        self.input_mean = torch.Tensor([0.3235, 0.3310, 0.3445])
        self.input_std = torch.Tensor([0.2533, 0.2224, 0.2483])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

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
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))
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

    def forward(self, x):
        
      #Implement a forward pass through the network, use forward for training
      #and detect for detection       

      print (self.detect(x))
      z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
      up_activation = []
      
      for i in range(self.n_conv):             #in range 4 basically.
        # Add all the information required for skip connections
        up_activation.append(z)
        z = self._modules['conv%d'%i](z)

      for i in reversed(range(self.n_conv)):
        z = self._modules['upconv%d'%i](z)
        # Fix the padding
        z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
        # Add the skip connection
        if self.use_skip:
          z = torch.cat([z, up_activation[i]], dim=1)
      return self.classifier(z)






#------------------------------------------------------------------------------>BEGIN  Detector.DETECT() METHOD FOR DETECTOR NETWORK
##=============================================================================>  USED BY Detector.FORWARD() METHOD ABOVE FOR DETECTION

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           
           image:           3 x H x W image
           return:          Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                            return no more than 30 detections per image per class. You only need to predict width and height
                            for extra credit. If you do not predict an object size, return w=0, h=0.
           
           Hint:  Use extract_peak here
           Hint:  Make sure to return THREE python lists of tuples of (float, int, int, float, float) and not a pytorch
                  scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                  out of memory.
        """
       
                  #extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
                
                    
                    #Extract local maxima (peaks) in a 2d heatmap.
                    #heatmap:               H x W heatmap containing peaks (similar to your training heatmap)
                    #max_pool_ks:           Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
                    #min_score:             Only return peaks greater than min_score
                    #return:                 List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                    #                        heatmap value at the peak. Return no more than max_det peaks per image
        
        
        print(extract_peak(image))
        return ("Hello World from Detector.Forward()--->detect() ****************OCT 29, 2021")              
   



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
