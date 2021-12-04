from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch
import string
from .utils import one_hot



def log_likelihood(model: LanguageModel, 
                   some_text: str):
   
    onehot_text = utils.one_hot(some_text)
    

    #how probable is the string some_text under this model?

    all_predictions = model.predict_all(some_text)
    #print ("\n Size of all_predictions is", all_predictions.shape) #([28, 7])
    all_predictions = all_predictions[:, :-1]  #remove last character prediction 
    #print ("\n New Size of all_predictions is", all_predictions.shape) #torch.Size([28, 6])
        
    likelihoods = all_predictions.t() @ onehot_text #multiply  one hot encoded text matrix by predictions matrix

    #this obtains the likelihood of the specific character at a specidic position
    #shape is len(some_text) x len(some_text) (e.g 6x6) 


    #print ("\n Size of one hot encoded text is ", text.shape) #([28, 6]))
    #print ("\n  Size of TRANSPOSE all_predictions is", all_predictions.t().shape) #([6, 28])
    #print ("\n  Size of likelihoods  is",likelihoods.shape) #([6, 6])
    
    
    output = likelihoods.diag()

    output = sum(output) #The log-likelihood is the sum of all individual likelihoods, not the average

    return  output


def sample_random(model: LanguageModel, 
                  max_length: int = 100):
    
    """
    https://piazza.com/class/ksjhagmd59d6sg?cid=1033
    
    >>> m =torch.distributions.categorical.Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
    https://pytorch.org/docs/stable/distributions.html
    
    CLASStorch.distributions.categorical.Categorical(probs=None, logits=None, validate_args=None)
    make sure use “logit = language_model_output”.
    
    --->LanguageModel class has a method of predict_next(), which will give you the distribution.
    ---->predict_next can take an empty string
    
    
    
    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    
    stringD = ""
    vocab = string.ascii_lowercase + ' .'
    
    for i in range (max_length):
      prob_next = model.predict_next(stringD)
      random_i = torch.distributions.categorical.Categorical(logits=prob_next).sample()
      stringD = stringD + vocab[random_i]
      if vocab[random_i] == '.':
        break
    
    return(stringD)



class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)

#############################################B  E   A  M   S E A R CH ###############
def beam_search(model: LanguageModel, 
                beam_size: int, 
                n_results: int = 10, 
                max_length: int = 100, 
                average_log_likelihood: bool = False):
   
# heap will sort by first element in tuple
    heap = TopNHeap(beam_size)
    term_heap = TopNHeap(n_results)
    visited = set()
    term_list = []
    strings_only = []
    

    # initialize heap
    prediction = model.predict_next("")  #shape [28]

    for i, likelihood in enumerate(prediction):  #i=0 to 27, and likelihood[i]
        
        visited.add(utils.vocab[i])  #iterate through all letters a to z and . (period)

        if utils.vocab[i] == '.':
            #print ("\n the Period corresponds to likelihood ---- ", l)
            term_heap.add(( likelihood, utils.vocab[i]) )
        else:
            #print ("\n this letter corresponds to likelihood that follows ---- ",  utils.vocab[i], l)
            heap.add( (likelihood, utils.vocab[i]) )

    m = 0

    while m < 40:

        for curr_likelihood, curr_s in heap.elements:

            prediction = model.predict_next(curr_s)

            for i, likelihood in enumerate(prediction):
                
                new_s = curr_s + utils.vocab[i]
                
                if average_log_likelihood:
                    new_likelihood = log_likelihood(model, new_s) / len(new_s)
                else:
                    new_likelihood = curr_likelihood + likelihood

                if new_s not in visited:
                    visited.add(new_s)
                    # add to terminated heap
                    if new_s[-1] == '.' or len(new_s) > max_length:
                        term_heap.add( (new_likelihood, new_s) )
                    else:
                        heap.add( (new_likelihood, new_s) )
        m += 1

    
    
    for i in range(len(term_heap.elements)):
        term_list.append(term_heap.elements[i])
    
    term_list.sort()

    
    #extract strings
    
    for likelihood, string in term_list:
        strings_only.append(string)
        print ("\nThis string being extracted from beam_search", string) 
    
    return strings_only




if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
