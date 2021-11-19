from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch
import string
from .utils import one_hot



def log_likelihood(model: LanguageModel, 
                   some_text: str):
    """
    

    NOV 19:  https://piazza.com/class/ksjhagmd59d6sg?cid=1075
    The log-likelihood is a single number, predict_all output an array of [28, n+1]. For a *single* 28-dim vector, 
    you will need only one of them that represents the current character.
          
    
    
    NOV 15: https://piazza.com/class/ksjhagmd59d6sg?cid=1033




   
    INSTRUCTIONS:

    This function takes a string as input and returns the log probability 
    of that string under the current language model. Test your implementation using the Bigram 
    or AdjacentLanguageModel.
    
    https://stackoverflow.com/questions/42817834/understanding-maximum-likelihood-in-nlp
    https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/
    It is common in optimization problems to prefer to minimize the cost function, rather than to maximize it.
    Therefore, the negative of the log-likelihood function is used, referred to generally as a
    Negative Log-Likelihood (NLL) function.
    
    https://moonbooks.org/Articles/How-to-calculate-a-log-likelihood-in-python-example-with-a-normal-distribution-/
    
    
    
    Hint: Remember that the language model can take an empty string as input

    Hint: Recall that LanguageModel.predict_all returns the log probabilities of the next characters for all substrings.

    Hint: The log-likelihood is the sum of all individual likelihoods, not the average

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    string = ""

    if some_text:                        #remove this for 20 points!
      string = some_text                 #remove this for 20 points!

    #print(f'Nov 19 2021 --------------- string len shape is {len(string)}')
    logit = model.predict_all(string)[:, :-1]
    v,i = torch.max(logit, dim=0)

    #print(f'Nov 19 2021 --------------- logit shape is {logit.shape}')

    LL = v.sum()

    return LL
    


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
    """
    
    Nov 19 2021


    https://piazza.com/class/ksjhagmd59d6sg?cid=1086

    Generate several sentences and try to find the top n sentences with highest log_likelihood.
    But since we have 28 choices for up to max_length characters, do we need to calculate  all
    the choices and then feed into the TopNHeap list?    



    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    Each column is a slot, there are up to max_legth such slots
    Rows are 0-27,  tells you how probable each letter is in each slot (column)


    slot:1   2  3   4

    letter

    a    .5  .2  .1  .9 
    b    .7  .4   0  .2
    c    .1  .6  .5  .4

    These are the highest values for the slots:

        1   2   3   4             1   2   3   4
       .7  .6  .5 .9              b   c   c   a    = sentence

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    vocab = string.ascii_lowercase + ' .'
    
    print (f'Inside of beam_search, returning list of strings of size n_results size: {n_results}')
    
    stringD = ""
    my_list = []
    my_heap = TopNHeap(beam_size)
    
    for i in range (n_results):
      for i in range (max_length):
        prob_next = model.predict_next(stringD)
        prob_all =  model.predict_all(stringD)
        max_i = torch.argmax(model.predict_next(stringD))
        stringD = stringD + vocab[max_i]
        my_heap.add(prob_next[max_i])   #MY HEAP HEAP
        if vocab[max_i] == '.':
          break
      
      my_list.append(stringD)
    
    
    #print(f'THE HEAP----------------> {my_heap.elements}')
    #print (max_length)

    #print ("-----------THE LIST----------------------------------------")
    #print (my_list)
    

    return (my_list)


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
