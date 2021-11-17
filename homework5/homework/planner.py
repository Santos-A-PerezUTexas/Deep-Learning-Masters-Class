import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class Planner(torch.nn.Module):


    def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
      
      super().__init__()
        
        
      c = n_input_channels    #3 in our case

      self.layer1 = torch.nn.Sequential(
      
            torch.nn.Conv2d(c, 32, kernel_size=5, stride=2, padding=5//2), 
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            
            )    


      self.layerUPCONV = torch.nn.Sequential(
      
            torch.nn.ConvTranspose2d(32, 16, kernel_size=5, padding=5 // 2, stride=2, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            
            )    


      self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=5//2), 
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            )          


      self.final = torch.nn.Linear(12288, 2)   #this takes 32x32 image of layer1 or 2, 32 channels
       
    
            
    def forward(self, img):
    
     


      #ADD A SKIP CONNECTION
      #ADD A SKIP CONNECTION
      #ADD A SKIP CONNECTION
      #ADD A SKIP CONNECTION
      
      #print(f'1    img shape is {img.shape}')
      
      out = self.layer1(img)
      #print(f'After Layer 1        out.shape is {out.shape}')
      
      out = self.layerUPCONV(out)
      #print(f'After UPCONV        out.shape is {out.shape}')
      
      out = self.layer2(out)
      #print(f'After Layer 2        out.shape is {out.shape}')
      
      argm = spatial_argmax(out[:,0 ,:, :])

      
      out = out.reshape(out.size(0), -1)
      out = self.final(out)
      
      #ARGMAX
              
      return argm








def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
