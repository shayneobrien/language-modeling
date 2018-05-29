import torchtext, random, math
import numpy as np
from collections import defaultdict
from math import log
from tqdm import tqdm
from collections import Counter

TEXT = torchtext.data.Field()
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".", train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

class TrigramModel:
    def __init__(self, train_iter, sample_size = 100, TEXT = TEXT, n = 3, oov_prob = 1e-10):
        self.n = n
        self.oov_prob = oov_prob
        self.TEXT = TEXT
        self.sample_size = sample_size
        self.probs_dict, self.alphas, self.perplexity = self.get_probs_dict(train_iter)
        
    def __call__(self, string):
        split_str = string.split()
        relevant_str = split_str[-1*self.n+1:]
        ids = tuple([model.TEXT.vocab.stoi[word] for word in relevant_str])

        all_probs = []
        for unigram in model.probs_dict[1].keys():
            combo = ids + unigram
            probability = 1
            for ngram in range(self.n, 0, -1):
                if combo[-ngram:] not in model.probs_dict[ngram]:
                    probability *= self.oov_prob
                else:
                    probability *= model.probs_dict[ngram][combo[self.n-ngram:]]
            all_probs.append(tuple([combo[-1], probability]))

        all_probs = sorted(all_probs, key = lambda x: x[1])
        out_ids = [prob[0] for prob in all_probs[-20:]]
        out_words = ' '.join([self.id_to_word(idx) for idx in out_ids])
        return out_words
            
    def get_probs_dict(self, train_iter):
        print('Finding best alpha values out of {0} random search values...'.format(self.sample_size))
        counts_dict = self.get_counts_dict(train_iter, self.n)
        ngrams = list(counts_dict.keys())[1:]
        best_ppl = 1e10
        for _ in tqdm(range(self.sample_size)):
            alphas = self.sample_alphas()
            probs_dict = defaultdict(dict)
            
            # retrieve probabilities
            for ngram in ngrams:
                probs_dict[ngram] = defaultdict(float)
                below_ngram = ngram-1
                for key, value in counts_dict[ngram].items():
                    below_key = key[:below_ngram]
                    probs_dict[ngram][key] = math.log(alphas[-ngram] * (value / counts_dict[below_ngram][below_key]), 2)

            # unigram is special case with this setup
            probs_dict[1] = {key: math.log(alphas[-1] * (value/sum(counts_dict[1].values()))) for key, value in counts_dict[1].items()}    
            ppl = self.perplexity(probs_dict, counts_dict, self.n)
            if ppl < best_ppl:
                best_ppl = ppl
                best_probs_dict = probs_dict
                best_alphas = alphas
                
        print('Best alphas: {1}'.format(best_ppl, best_alphas))
            
        return best_probs_dict, best_alphas, best_ppl

    def get_counts_dict(self, train_iter, n):
        # initialize dictionary of ngram dictionaries
        counts_dict = defaultdict(dict)
        interval = range(1, n+1)
        for n_val in interval: 
            counts_dict[n_val] = defaultdict(int)

        # get all ngram counts, store
        for batch in iter(train_iter):
            generators = [self.collect_batch_ngrams(batch, n) for n in interval]
            for n_val, gen in enumerate(generators):
                for entry in gen:
                    counts_dict[n_val+1][entry] += 1

        return counts_dict

    def collect_batch_ngrams(self, batch, n):
        n = max(1, int(n))
        data = batch.text.view(-1).data.tolist()
        for idx in range(0, len(data)-n+1):
            yield tuple(data[idx:idx+n])

    def sample_alphas(self):
        alpha1 = random.random()
        alpha2 = random.uniform(0, 1-alpha1)
        return [alpha1, alpha2, 1-alpha1-alpha2]
    
    def perplexity(self, probs_dict, counts_dict, n = 3):
        average_nll = np.mean([-probs_dict[n][ngram] for ngram in probs_dict[n].keys()])
        return np.exp(average_nll)
    
    def id_to_word(self, idx):
        return self.TEXT.vocab.itos[idx]
    
    def write_kaggle(self, input_file):
        print('Writing output...')
        inputs = open(input_file, 'r').read().splitlines()
        with open('trigram_output.txt', 'w') as fh:
            fh.write('id,word\n')
            for idx, line in enumerate(tqdm(inputs)):
                fh.write(str(idx) + ',' + self(line[:-4]) + '\n')

model = TrigramModel(train_iter, sample_size = 25)
model.write_kaggle('input.txt')

