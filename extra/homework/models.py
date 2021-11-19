import torch
import torch.nn.functional as F

from . import utils
from .utils import one_hot


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


    #bigram predictALL

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

#==============================================TCN===========================================



class TCN(torch.nn.Module, LanguageModel):     #MY WARNING:  TCN in example DOES NOT inherit from Language Model
    class CausalConv1dBlock(torch.nn.Module):
        
        
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
          
            self.pad1d = torch.nn.ConstantPad1d((2*dilation,0), 0)
            self.c1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, dilation=total_dilation)
            #self.b1 = torch.nn.BatchNorm2d(out_channels)               
         
        def forward(self, x):
            return F.relu(self.c1(self.pad1d(x)))

    
    
    
    
    #--------------->TCN INIT

    def __init__(self, layers=[28,16,8], char_set="string"):   #<---------------------------added char_set 11/16/2021
        
        """
       
        Your TCN model will use a CausalConv1dBlock. This block combines causal 1D convolution with a non-linearity (e.g. ReLU).
        The main TCN then stacks multiple dilated CausalConv1dBlock’s to build a complete model. Use a 1x1 convolution to produce the output.
        TCN.predict_all should use TCN.forward to compute the log-probability from a single sentence.

        Hint: Make sure TCN.forward uses batches of data.  <---------------------------*******

        Hint: Make sure TCN.predict_all returns log-probabilities, not logits.

        Hint: Store the distribution of the first character as a parameter of the 
        model torch.nn.Parameter

        Hint: Try to keep your model manageable and small. The master solution trains
         in 15 min on a GPU.


        http://www.philkr.net/dl_class/lectures/sequence_modeling/07.html
        
        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        
        
        ***-->Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        
        torch.nn.Parameter
        torch.nn.functional.log_softmax
        torch.nn.ConstantPad1d
        
        
        https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter
        
        class NN_Network(nn.Module):
            def __init__(self,in_dim,hid,out_dim):
                 super(NN_Network, self).__init__()
                 self.linear1 = nn.Linear(in_dim,hid)
                 self.linear2 = nn.Linear(hid,out_dim)
                 self.linear1.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
                 self.linear1.bias = torch.nn.Parameter(torch.ones(hid))
                 self.linear2.weight = torch.nn.Parameter(torch.zeros(in_dim,hid))
                 self.linear2.bias = torch.nn.Parameter(torch.ones(hid))
    
        """
        
        super().__init__()
        
        #ADD PROBABILITY AS PARAMETER HERE<------------------
        
        #self.conv.weight = torch.nn.Parameter(weight)
        
        prob = torch.nn.Parameter(torch.zeros(32, 28))
        #print (f'shape of prob is {prob.shape}')

        c = 28 

        print(f'------------->length of char set is {c}')

        L = []
        total_dilation = 1
        for l in layers:
            L.append(torch.nn.ConstantPad1d((2*total_dilation,0), 0))
            L.append(torch.nn.Conv1d(c, l, 3, dilation=total_dilation))
            L.append(torch.nn.ReLU())
            total_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, 28, 1)
        

    #--------------------------TCN FORWARD()
    #--------------------------TCN FORWARD()
    #--------------------------TCN FORWARD()
    
    def forward(self, x):  #x is a sequence of any length

        """
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings, (32,28, L)
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits

        Crash "Given groups=1, weight of size [8, 6, 3], expected input[1, 28, 102] to have 6 channels,
         but got 28 channels instead

        """
        
        #print("-----------------------------------------------------------------")
        #print ("In FORWARD()")
        #print("-----------------------------------------------------------------")
        
        #print(f'Nov 19, shape of x is {x.shape}')

        self.prob = x[:, :, 0]

        #print(f'                     the self.prob shape is {self.prob.shape}')
        output = self.network(x)
        
        #print(f'Nov 19, shape of first output is {output.shape}')

        output = self.classifier(output) 
        
        #print(f'Nov 19, shape of CLASSIFICATION is {output.shape}')
        #print ("END FORWARD()")
        #print("-----------------------------------------------------------------")
        
        #output = output + torch.cat((output, self.prob[0,:]), dim=0)
        
        #print(f'Nov 200000, shape of second output is {output.shape}')

        
        return    output  # shape ([128, 29, 256]), 128 batches, 29 character alphabet or vocab_size, 256 letters in the string
        
        
        
        

    def predict_all(self, some_text):
        """
        Your code here

        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        
        RETURN  (28,  L+1)  - Log Likelikelyhoods

        """
        
        one_hotx = one_hot(some_text)

        print (f'in predict_all, sometext is {some_text}')
        print (f'in predict all one_hotx shape is {one_hotx.shape}')

        output = self.forward(one_hotx)

        print (f'in predict all output shape is {output.shape}')
        
        return(output)

        
        
        
        
        
        
        
        

def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
