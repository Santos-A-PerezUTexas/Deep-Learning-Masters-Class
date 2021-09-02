from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



"""
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY 
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY
UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY UTILS.PY  UTILS.PY


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path)
    def __len__(self):
    def __getitem__(self, idx): 



def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
    

"""

"""

https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

This change made DIRECTLY on Github.  THIS change 9/1/2021

As a first step, we will need to implement a data loader for the SuperTuxKart dataset. Complete the __init__, __len__, and the __getitem__ of the SuperTuxDataset class in the utils.py.
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


The __len__ function should return the size of the dataset.

The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.

Labels and the corresponding image paths are saved in labels.csv, their headers are *file* and *label*. 
There are 6 classes of objects. Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.


Hint: We recommend using the csv package to read csv files and the PIL library (Pillow fork) to read images in Python.

Hint: Use torchvision.transforms.ToTensor() to convert the PIL image to a pytorch tensor.

Hint: You have (at least) two options on how to load the dataset. 
You can load all images in the __init__ function, or you can lazily load them in __getitem__.
If you load all images in __init__, make sure you convert the image to a tensor in the constructor, otherwise, you might get an OSError: [Errno 24] Too many open files.


"""


LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        
        You can load all images in the __init__ function, or you can lazily load them in __getitem__.
        If you load all images in __init__, make sure you convert the image to a tensor in the constructor, otherwise, 
        you might get an OSError: [Errno 24] Too many open files
        
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        
        Your code here
        *****Hint: Use the python csv library to parse labels.csv
        https://docs.python.org/3/library/csv.html#module-csv

       ****Were is labels.csv???????????????????????

        WARNING: Do not perform data normalization here. 
        
        STDataset = pd.read_csv('data/labels.csv')

        
        """
        
        #raise NotImplementedError('SuperTuxDataset.__init__')

        
        """               SAMPLE CODE*************************************
        landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

         n = 65
        img_name = landmarks_frame.iloc[n, 0]
        landmarks = landmarks_frame.iloc[n, 1:]
        landmarks = np.asarray(landmarks)
        landmarks = landmarks.astype('float').reshape(-1, 2)

        print('Image name: {}'.format(img_name))
        print('Landmarks shape: {}'.format(landmarks.shape))
        print('First 4 Landmarks: {}'.format(landmarks[:4]))
        
        """
        
        
        
        
        
        
        
        
    def __len__(self):
        """
        
        The __len__ function should return the size of the dataset.
        
        Your code here
        """
        raise NotImplementedError('SuperTuxDataset.__len__')


        
        
    def __getitem__(self, idx):     
        """
        Your code here
        
        __getitem__ to support the indexing such that dataset[idx] can be used to get idx-th sample.
        
        
        You can load all images in the __init__ function, or you can lazily load them in __getitem__.
        
        return------> a tuple: img, label   #https://www.freecodecamp.org/news/python-returns-multiple-values-how-to-return-a-tuple-list-dictionary/

        Labels and the corresponding image paths are saved in labels.csv, their headers are *file* and *label*. 
        img is a tensor???????
        Pil image to tensor:  https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312
        
            print("t is: ", t.size()) 
            from torchvision import transforms
            img = transforms.ToPILImage()(t).convert("RGB")
            display(img)
            print(img)
            print(img.size)
        
        
        
        
        def person():
            return "bob", 32, "boston"
   
        https://stackoverflow.com/questions/62660486/using-image-label-dataset-take2-returns-two-tuples-instead-of-a-single-one
        
        
        The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.
        
        """
        raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
