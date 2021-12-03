import torch
import torch.nn.functional as F

from . import utils
from .utils import one_hot


#NOTE THIS IS JUST TO TEST THE GRADER, NOV 15 2021


class LanguageModel(object):
    
    def predict_all(self, some_text):
        """
      
        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        
    def predict_next(self, some_text):
        """
        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
 
    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))


    #bigram predictALL

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
 
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
        
               
        super().__init__()
        
       
        c = 28 
        
        L = []
        total_dilation = 1 # starting dilation at 2, not 1?????
        for l in layers:
            L.append(torch.nn.ConstantPad1d((2*total_dilation,0), 0))
            L.append(torch.nn.Conv1d(c, l, kernel_size=3, dilation=total_dilation))
            L.append(torch.nn.ReLU())
            total_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, 28, 1)
        
    #--------------------------TCN FORWARD()
    
    def forward(self, x):  #x is a sequence of any length

        """
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings, (32,28, L)
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits

        """
        first_char_distribution = torch.nn.Parameter(torch.rand(x.shape[0], x.shape[1], 1))

        #this EXPLAINS IT ALL!!!!!!!!!!!!!!:  NOV 28 2021
        #https://piazza.com/class/ksjhagmd59d6sg?cid=1229
        #https://piazza.com/class/ksjhagmd59d6sg?cid=1229
        #Then concatenate it with the output of TCN.
        #Then concatenate it with the output of TCN.
        #The P( next | “”) is the parameter you defined, and will learn from the data later.
        #x = torch.cat((x,self.param),dim=2)
        #print(f'Nov 21, the NEW shape of x is {x.shape}')   #([32, 28, 1])

        output = self.network(x)  #x is  [32,28, L]
        output = self.classifier(output)

        output = torch.cat((first_char_distribution, output),dim=2) 
        #first_char_distribution is [32, 28, 1]
        
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
        
        #one_hotx = one_hot(some_text)[:, :-1]

        one_hotx = one_hot(some_text)[None]

        print (f'Dec 2 in predict_all, sometext is {some_text}')
        print (f'Dec 2 in predict all one_hotx shape is {one_hotx.shape}')

        output = self.forward(one_hotx)

        print (f'in predict all output shape is {output.shape}')
        
        return(output.log())

        
        
        
        
        
        
        
        

def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
