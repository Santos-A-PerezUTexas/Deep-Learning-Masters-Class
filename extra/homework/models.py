import torch
import torch.nn.functional as F

from . import utils

#NOTE THIS IS JUST TO TEST THE GRADER, NOV 15 2021


class LanguageModel(object):
    
    def predict_all(self, some_text):
        """
        The main task of the character-level language model is to predict the next character 
        given all previous characters in a sequence of data, i.e. generates text character by character.
        
        https://towardsdatascience.com/character-level-language-model-1439f5dd87fe
        
        https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
        
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        NOT PART OF EXTRA CREDIT, SEPT 21
        
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
    """
    
    IMPLEMENTED BELOW IN TH FILE. SEPT 21
    
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()


class TCN(torch.nn.Module, LanguageModel):     #MY WARNING:  TCN in example DOES NOT inherit from Language Model
    class CausalConv1dBlock(torch.nn.Module):
        
        
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            """
            Your code here.
            Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
            Optionally, repeat this pattern a few times and add in a residual block
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            self.pad1d = torch.nn.ConstantPad1d((2*dilation,0), 0)
            self.c1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, dilation=total_dilation)
            #self.b1 = torch.nn.BatchNorm2d(out_channels)
                        
            
        def forward(self, x):
            return F.relu(self.c1(self.pad1d(x)))

    
    
    
    
    
    def __init__(self, layers=[32,64,128,256]):   #added layers 11/16/2021
        
        """
        Your code here  (BEGIN TCN CONSTRUCTOR)
        
        http://www.philkr.net/dl_class/lectures/sequence_modeling/07.html
        
        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
        
        super().__init__()
        c = len(char_set)
        L = []
        total_dilation = 1
        for l in layers:
            L.append(torch.nn.ConstantPad1d((2*total_dilation,0), 0))
            L.append(torch.nn.Conv1d(c, l, 3, dilation=total_dilation))
            L.append(torch.nn.ReLU())
            total_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, len(char_set), 1)
        
    def forward(self, x):
        """
        Your code here
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        return self.classifier(self.network(x))
        
        
        
        

    def predict_all(self, some_text):
        """
        Your code here

        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        raise NotImplementedError('TCN.predict_all')

        
        
        
        
        
        
        
        

def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
