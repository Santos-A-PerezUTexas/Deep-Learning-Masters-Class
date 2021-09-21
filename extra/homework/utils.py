import re
import string

import torch
from torch.utils.data import Dataset

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
        st = st.lower()
        reg = re.compile('[^%s]' % vocab)
        period = re.compile(r'[ .]*\.[ .]*')
        space = re.compile(r' +')
        sentence = re.compile(r'[^.]*\.')
        self.data = space.sub(' ',period.sub('.',reg.sub('', st)))
        if max_len is None:
            self.range = [(m.start(), m.end()) for m in sentence.finditer(self.data)]
        else:
            self.range = [(m.start(), m.start()+max_len) for m in sentence.finditer(self.data)]
            self.data += self.data[:max_len]
        if transform is not None:
            self.data = transform(self.data)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, idx):
        s, e = self.range[idx]
        if isinstance(self.data, str):
            return self.data[s:e]
        return self.data[:, s:e]


if __name__ == "__main__":
    data = SpeechDataset('data/valid.txt', max_len=None)
    print('Dataset size ', len(data))
    for i in range(min(len(data), 10)):
        print(data[i])

    data = SpeechDataset('data/valid.txt', transform=one_hot, max_len=None)
    print('Dataset size ', len(data))
    for i in range(min(len(data), 3)):
        print(data[i])
