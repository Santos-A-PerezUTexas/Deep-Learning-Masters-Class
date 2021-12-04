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
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()

        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]

        h, conv_layers = 3, []

        for c in channels:
            conv_layers += conv_block(c, h)
            h = c

        self.conv_layers = torch.nn.Sequential(*conv_layers)
        
        #self.linear_classifier = torch.nn.Linear(475, 1)
        self.aimpoint_classifier = torch.nn.Conv2d(h, 1, 1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        
        #print ("\n\n coordinates[0,:,:,:] view -1 shape is ", coordinates[0,:,:,:].view(-1).shape) 
        #print ("\n\n coordinates[0,:,:,:]  shape is ", coordinates[0,:,:,:].shape)

        #flag = self.classifier(coordinates.mean(dim=[-2, -1]))  #added Dec 3, 2021
        #print ("\n\n flag shape is ", flag.shape)

        coordinates = self.conv_layers(img)

        #print ("\n\n 0- coordinates  shape after network is ", coordinates.shape)

        coordinates = self.aimpoint_classifier(coordinates)
        #print ("\n\n 1 coordinates  shape after classifier is ", coordinates.shape)

        coordinates = spatial_argmax(coordinates[:, 0])
        #print ("\n\n 2 coordinates  after argmax shape is ", coordinates.shape)


        return coordinates, 1  #added Dec 3, 2021

        #return output  #deleted dec 3
        
        # return self.classifier(coordinates.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    print ("--------------------------------IN LOAD MODEL")
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '/content/cs342/final/image_agent/planner.th'), map_location='cpu'))
    return r
