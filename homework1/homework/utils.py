from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


"""

This change made DIRECTLY on Github.  THIS change 9/1/2021

As a first step, we will need to implement a data loader for the SuperTuxKart dataset. Complete the __init__, __len__, and the __getitem__ of the SuperTuxDataset class in the utils.py.

The __len__ function should return the size of the dataset.

The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.

Labels and the corresponding image paths are saved in labels.csv, their headers are file and label. There are 6 classes of objects. Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.


Hint: We recommend using the csv package to read csv files and the PIL library (Pillow fork) to read images in Python.

Hint: Use torchvision.transforms.ToTensor() to convert the PIL image to a pytorch tensor.

Hint: You have (at least) two options on how to load the dataset. You can load all images in the __init__ function, or you can lazily load them in __getitem__. If you load all images in __init__, make sure you convert the image to a tensor in the constructor, otherwise, you might get an OSError: [Errno 24] Too many open files.


"""


LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
