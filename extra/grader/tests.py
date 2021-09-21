"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
import torch
import numpy as np
import string

from .grader import Grader, Case, MultiCase

vocab = string.ascii_lowercase+' .'


def one_hot(s: str):
    if len(s) == 0:
        return torch.zeros((len(vocab), 0))
    return torch.as_tensor(np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]).float()


class LanguageGrader(Grader):
    """Language modeling"""

    def __init__(self, *a, **ka):
        super().__init__(*a, **ka)
        self.bigram = self.module.Bigram()

        class Dummy(self.module.LanguageModel):
            def predict_all(self, text):
                r = 1e-5*torch.ones(len(vocab), len(text)+1)
                for i, s in enumerate(text):
                    r[(vocab.index(s)+1)%len(vocab), i+1] = 1
                r[0, 0] = 2
                r[1, 0] = 1
                return (r/r.sum(dim=0, keepdim=True)).log()

        self.dummy = Dummy()

    @Case(score=10)
    def test_log_likelihood(self):
        """log_likelihood"""
        ll = self.module.log_likelihood

        def check(m, s, r):
            l = ll(m, s)
            assert abs(r-l) < 1e-2, "wrong log likelihood for '%s' got %f expected %f"%(m.__class__.__name__, l, r)

        check(self.bigram, 'yes', -8.914730)
        check(self.bigram, 'we', -3.708824)
        check(self.bigram, 'can', -7.696493)
        check(self.dummy, 'abcdef', -0.406903)
        check(self.dummy, 'abcdee', -11.919827)
        check(self.dummy, 'bcdefg', -1.100051)

        def check_sum(m, length):
            all_str = []
            all_sub_str = ['']
            while len(all_sub_str):
                s = all_sub_str.pop()
                if len(s) == length:
                    all_str.append(s)
                else:
                    for c in vocab:
                        all_sub_str.append(s + c)

            l = np.sum([np.exp(ll(m, s)) for s in all_str])

            assert abs(1-l) < 1e-2, "Log likelihood for '%s' does not sum to 1" % (m.__class__.__name__)

        check_sum(self.bigram, length=0)
        check_sum(self.dummy, length=0)
        check_sum(self.bigram, length=1)
        check_sum(self.dummy, length=1)
        check_sum(self.bigram, length=2)
        check_sum(self.dummy, length=2)


    @Case(score=10)
    def test_sample_random(self):
        """sample_random"""
        ll = self.module.log_likelihood
        sample = self.module.sample_random

        def check(m, min_likelihood):
            samples = [sample(m) for i in range(10)]
            sample_ll = np.median([float(ll(m, s))/len(s) for s in samples])

            assert sample_ll > min_likelihood, \
                "'%s' : Samples should have a likelihood of at least %f got %f"%(m.__class__.__name__, min_likelihood,
                                                                                 sample_ll)

        check(self.bigram, -2.5)
        check(self.dummy, -0.05)

        samples = [sample(self.dummy) for i in range(100)]
        chars = [s[10] if len(s) > 10 else 'a' for s in samples]
        # There is a 1 in 100 billion chance this fill fail
        assert any([abs(sum([c == 'k' for c in chars[i: i+10]])-6.666) < 2 for i in range(0, 100, 10)]), \
            "Your samples seem biased"
        # There is a 1 in 100 billion chance this fill fail
        assert any([abs(sum([c == 'l' for c in chars[i: i+10]])-3.333) < 2 for i in range(0, 100, 10)]), \
            "Your samples seem biased"

    @Case(score=20)
    def test_beam_search(self):
        """beam_search"""
        ll = self.module.log_likelihood
        bs = self.module.beam_search

        def check(m, n, min_log_likelihood, average_log_likelihood=False):
            samples = bs(m, 100, n, max_length=30, average_log_likelihood=average_log_likelihood)
            assert len(samples) == n, "Beam search returned %d samples expected %d!"%(len(samples), n)
            assert all([s not in samples[:i] for i, s in enumerate(samples)]), 'Beam search returned duplicates'
            med_ll = np.median([float(ll(m, s))*(1./len(s) if average_log_likelihood else 1.) for s in samples])
            assert med_ll > min_log_likelihood, "Beam search failed to find high likelihood samples"

        check(self.bigram, 10, -7.5, False)
        check(self.bigram, 10, -1.5, True)
        check(self.dummy, 10, -12., False)
        check(self.dummy, 10, -0.4, True)
        check(self.dummy, 2, -0.8, False)
        check(self.dummy, 2, -0.1, True)


class TCNGrader(Grader):
    """TCN"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tcn = self.module.TCN()
        self.tcn.eval()

    @MultiCase(score=3, b=[1, 16, 128], length=[0, 10, 20])
    def test_forward(self, b, length):
        """TCN.forward"""
        one_hot = (torch.randint(len(vocab), (b, 1, length)) == torch.arange(len(vocab))[None, :, None]).float()
        output = self.tcn(one_hot)
        assert output.shape == (b, len(vocab), length+1), \
            'TCN.forward output shape expected (%d, %d, %d) got %s'%(b, len(vocab), length+1, output.shape)

    @MultiCase(score=3, s=['', 'a', 'ab', 'abc'])
    def test_predict_all(self, s):
        """TCN.predict_all"""
        output = self.tcn.predict_all(s)
        assert output.shape == (len(vocab), len(s)+1), \
            'output shape expected (%d, %d) got %s' % (len(vocab), len(s)+1, output.shape)
        assert np.allclose(output.exp().sum(dim=0).detach(), 1), "log likelihoods do not sum to 1"

    @MultiCase(score=7, s=['united', 'states', 'yes', 'we', 'can'])
    def test_consistency(self, s):
        """TCN.predict_next/TCN.predict_all consistency"""
        all_ll = self.tcn.predict_all(s).detach().cpu().numpy()
        for i in range(len(s)+1):
            ll = self.tcn.predict_next(s[:i]).detach().cpu().numpy()
            assert np.allclose(all_ll[:, i], ll), \
                "predict_next %s inconsistent with predict_all %s"%(ll, all_ll[:, i])

    @MultiCase(score=7, i=range(100))
    def test_causal(self, i):
        """TCN.forward causality"""
        input = torch.zeros(len(vocab), 100)
        input[:, i] = float('NaN')
        output = self.tcn(input[None])[0]
        is_nan = (output != output).any(dim=0)
        assert not is_nan[:i+1].any(), "Model is not causal, information leaked forward in time"
        assert is_nan[i+3:].any(), "Model does not consider a temporal extend > 2"

    @MultiCase(score=2, i=range(5,95))
    def test_shape(self, i):
        """TCN.forward shape"""
        input = torch.zeros(len(vocab), i)
        output = self.tcn(input[None])[0]
        assert (output.shape[0] == input.shape[0]) and (output.shape[1] == input.shape[1]+1), "Expected output shape (%d, %d) for input shape (%d, %d)!" % (input.shape[0], input.shape[1]+1, input.shape[0], input.shape[1])

    @MultiCase(score=5, i=range(10,90))
    def test_causal(self, i):
        """TCN.forward causality"""
        input = torch.zeros(len(vocab), 100)
        input[:, i] = float('NaN')
        output = self.tcn(input[None])[0]
        is_nan = (output != output).any(dim=0)
        assert not is_nan[:i+1].any(), "Model is not causal, information leaked forward in time"
        assert is_nan[i+3:].any(), "Model does not consider a temporal extend > 2"

class TrainedTCNGrader(Grader):
    """TrainedTCN"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tcn = self.module.load_model()
        self.tcn.eval()
        self.data = self.module.SpeechDataset('data/valid.txt')

    @Case(score=40)
    def test_nll(self):
        """Accuracy"""
        lls = []
        for s in self.data:
            ll = self.tcn.predict_all(s)
            lls.append(float((ll[:, :-1]*one_hot(s)).sum()/len(s)))
        nll = -np.mean(lls)
        return max(2.3-max(nll, 1.3), 0), 'nll = %0.3f' % nll
