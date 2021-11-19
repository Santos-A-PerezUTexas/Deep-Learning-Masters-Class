import re
import string

import torch
from torch.utils.data import Dataset, DataLoader

vocab = string.ascii_lowercase + ' .'


def one_hot(s: str):
    """
    Converts a string into a one-hot encoding
    :param s: a string with characters in vocab (all other characters will be ignored!)
    :return: a once hot encoding Tensor r (len(vocab), len(s)), with r[j, i] = (s[i] == vocab[j])
    """
    import numpy as np
    if len(s) == 0:
        return torch.zeros((len(vocab), 0))
    return torch.as_tensor(np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]).float()


class SpeechDataset(Dataset):
    """
    Creates a dataset of strings from a text file.
    All strings will be of length max_len and padded with '.' if needed.

    By default this dataset will return a string, this string is not directly readable by pytorch.
    Use transform (e.g. one_hot) to convert the string into a Tensor.
    """

    def __init__(self, dataset_path, transform=None, max_len=250):
        with open(dataset_path) as file:
            st = file.read()
        
        #code below seems to just make text lower case, remove spacing after period.  Tokenize?
        st = st.lower()
        reg = re.compile('[^%s]' % vocab)
        period = re.compile(r'[ .]*\.[ .]*')
        space = re.compile(r' +')
        sentence = re.compile(r'[^.]*\.')
        self.data = space.sub(' ',period.sub('.',reg.sub('', st)))  #removes spacing after period?
        
        if max_len is None:
            self.range = [(m.start(), m.end()) for m in sentence.finditer(self.data)]
        else:
            self.range = [(m.start(), m.start()+max_len) for m in sentence.finditer(self.data)]
            self.data += self.data[:max_len]

        if transform is not None:
            print("--------------------------------------------------------------------")
            print("Applying the one hot transform")
            print("--------------------------------------------------------------------")
            print("--------------------------------------------------------------------")
            print("Applying the one hot transform")
            print("--------------------------------------------------------------------")
            
            self.data = transform(self.data)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, idx):
        s, e = self.range[idx]
        #print (f'index is {idx}, s is {s}, and e is {e}')
        #print (f'self.range[idx] is {self.range[idx]}, and e-s is {e-s}')
        #index is 752, s is 79351, and e is 79418
        if isinstance(self.data, str):
            #print (f'Based on above numbers, returning self.data[s:e]: {self.data[s:e]} ')
            return self.data[s:e]        #if it's a string, return s to e substring only
       # print ("NOT A STRING")
        #print (f'returning self.data[:, s:e]  : {self.data[:, s:e].shape}')
        #shape of self.data[:, s:e] is ([28, s-e])
        return self.data[:, s:e]

def load_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = SpeechDataset(dataset_path, **kwargs)
    print("load_data()------->LOADED DATASET SANTOS, this one below:")
    print(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == "__main__":
    print ("Starting execution of utils.py")
    
    """
    data = SpeechDataset('data/valid.txt', max_len=None)
    print('Dataset size BEFORE TRANSFORM is ', len(data))

    print (f'data[0] is {data[0]}')
    print (f'data[1] is {data[1]}')  #is i did not know mr.
    print (f'data[2] is {data[2]}')  #is cronkite personally.
    print (f'data[1][x] is {data[1][14]}')
    print (f'data[2][x] is {data[2][15]}')

    
    print (f'data[0] size is {len(data[0])}')
    print (f'data[1] size is {len(data[1])}')  #data[1] size is 18
    print (f'data[2] size is {len(data[2])}')  #data[2] size is 20

    #for i in range(min(len(data), 10)):
     #   print(data[i])

    print("-------------------------------")
    
    #print("From main()------>tranforming data to one hot now")

    data = SpeechDataset('data/valid.txt', transform=one_hot, max_len=None)
    
    print (f'data[0] size is {data[0].shape}')
    print (f'data[1] size is {data[1].shape}')
    print (f'data[2] size is {data[2].shape}')

    print('TRANSFORMED Dataset size ', len(data))  #856
    #print('TRANSFORMED Dataset shape ', data.dtype)  
    
    #for i in range(min(len(data), 3)):
     #   print(data[i])

     """

    print("--------------------------------------------------------------------")
    print ("USING THE LOADER NOW")
    train_data = load_data('data/train.txt',  transform=one_hot)
    
    #train_data = load_data('data/valid.txt',  max_len=None)
    #Transform did not work, raised stack exception: load_data('data/valid.txt',  transform=one_hot, max_len=None)  
    for s in train_data:
      print (s[0])#torch.Size([32, 28, 250])
