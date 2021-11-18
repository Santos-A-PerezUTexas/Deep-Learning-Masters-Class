from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils


def log_likelihood(model: LanguageModel, some_text: str):
    """
    
    NOV 15: https://piazza.com/class/ksjhagmd59d6sg?cid=1033

    This case is like you have a bag of balls in different colors (characters), you know the distribution,
    like there are 3 red balls ®, 4 green balls (G), 5 blue ball (G) in the bag. The prob_next_ball = [3/12, 4/12, 5/12]

    Then you keep drawing balls, write down the color of the ball, and put it back.

    The balls are like characters, and finally you have a sequence like R G B B G R …

    The prob_next_char will come from the language model.

    Please see the example of “Categorical” in the following link, to see how to sample (draw) a number
    based on a given distribution (given by the language model)

    https://pytorch.org/docs/stable/distributions.html


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
    
    
    
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    
    return (-8.914730)


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    return("Returned from Sample Random")



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


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """

    stuff = ["apple", "banana", "cherry", "Crypto", "Bitcoin", "Music", "Tesla", "cars", "plane", "berry" ]

    my_list = []

    for i in range (n_results):
      my_list.append(stuff[i])

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
