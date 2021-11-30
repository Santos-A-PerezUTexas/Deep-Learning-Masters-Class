from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch
import string
from .utils import one_hot



def log_likelihood(model: LanguageModel, 
                   some_text: str):
   
    text = utils.one_hot(some_text)
    size_of_string = len(some_text)
    all_predictions = model.predict_all(some_text)
    print ("\n Size of all_predictions is", all_predictions.shape)
    all_predictions = all_predictions[:, 0:size_of_string]
    print ("\n New Size of all_predictions is", all_predictions.shape)
    
    likelihoods = torch.mm(all_predictions.t(), text)   #multiply text one hot encoded by predictions
    
    print ("\n Size of one hot encoded text is ", text.shape)
    print ("\n  Size of TRANSPOSE all_predictions is", all_predictions.t().shape)
    
    


    output = sum(likelihoods.diag()) #The log-likelihood is the sum of all individual likelihoods, not the average

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

    # initialize heap
    likelihoods = model.predict_next("")
    for i, l in enumerate(likelihoods):
        visited.add(utils.vocab[i])
        if utils.vocab[i] == '.':
            term_heap.add(( l, utils.vocab[i]) )
        else:
            heap.add( (l, utils.vocab[i]) )

    iters = 0
    while iters < 50:

        for curr_l, curr_s in heap.elements:
            likelihoods = model.predict_next(curr_s)

            for i, l in enumerate(likelihoods):
                new_s = curr_s + utils.vocab[i]
                if average_log_likelihood:
                    new_l = log_likelihood(model, new_s) / len(new_s)
                else:
                    new_l = curr_l + l

                if new_s not in visited:
                    visited.add(new_s)
                    # add to terminated heap
                    if new_s[-1] == '.' or len(new_s) > max_length:
                        term_heap.add( (new_l, new_s) )
                    else:
                        heap.add( (new_l, new_s) )
        iters += 1

    # sort and return
    sort_list = []
    for i in range(len(term_heap.elements)):
        sort_list.append(term_heap.elements[i])
    #sort_list.sort(reverse=True)
    sort_list.sort()

    return_list = []
    for l, s in sort_list:
        return_list.append(s)
    print(return_list)
    return return_list
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
