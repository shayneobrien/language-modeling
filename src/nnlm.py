import torchtext, random, torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1

TEXT = torchtext.data.Field()
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".", train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=1000) if False else TEXT.build_vocab(train)
TEXT.vocab.load_vectors('glove.840B.300d')
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test), batch_size=10, device=device, bptt_len=32, repeat=False)

class LanguageModel(nn.Module):
    """ neural network language model as per Bengio et al. 2013 """     
    def __init__(self, hidden_dim = 100, TEXT = TEXT):
        super(LanguageModel, self).__init__()
        
        vocab_size, embedding_dim = TEXT.vocab.vectors.shape
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.input2linear = nn.Linear(2*embedding_dim, hidden_dim)
        self.linear2output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embeddings(x)
        x_1 = x.view(len(x), -1)
        x_2 = F.tanh(self.input2linear(x_1))
        x_3 = self.linear2output(x_2)
        logits = F.log_softmax(x_3, dim = 1)        
        return logits
    
    def predict(self, x, TEXT = TEXT):
        embedded = self.embeddings(x)
        embedded = embedded.view(-1, 1).transpose(0,1)
        activated = F.tanh(self.input2linear(embedded))
        output = self.linear2output(activated)
        logits = F.log_softmax(output, dim = 1)
        out_ids = np.argsort(logits.data[0].tolist())[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words

class Trainer:
    def __init__(self, train_iter, val_iter):
        self.train_iter = train_iter
        self.val_iter = val_iter
        
    def string_to_batch(self, string):
        relevant_split = string.split()[-2:] # last two words, ignore ___
        ids = [self.word_to_id(word) for word in relevant_split]
        if use_cuda:
            return Variable(torch.LongTensor(ids)).cuda()
        else:
            return Variable(torch.LongTensor(ids))
        
    def word_to_id(self, word, TEXT = TEXT):
        return TEXT.vocab.stoi[word]
    
    def batch_to_input(self, batch):
        ngrams = self.collect_batch_ngrams(batch)
        x = Variable(torch.LongTensor([ngram[:-1] for ngram in ngrams]))
        y = Variable(torch.LongTensor([ngram[-1] for ngram in ngrams]))
        if use_cuda:
            return x.cuda(), y.cuda()
        else:
            return x, y
    
    def collect_batch_ngrams(self, batch, n = 3):
        data = batch.text.view(-1).data.tolist()
        return [tuple(data[idx:idx + n]) for idx in range(0, len(data) - n + 1)]
    
    def train_model(self, model, num_epochs):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr=1e-3)
        criterion = nn.NLLLoss()
        
        for epoch in tqdm(range(num_epochs)):
            
            model.train()
            epoch_loss = []
            
            for batch in self.train_iter:
                x, y = self.batch_to_input(batch)

                optimizer.zero_grad()

                y_pred = model(x)

                loss = criterion(y_pred, y)
                loss.backward()

                optimizer.step()

                epoch_loss.append(loss.data[0])
                
            model.eval()
            train_ppl = np.exp(np.mean(epoch_loss))
            val_ppl = self.validate(model)

            print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, np.mean(epoch_loss), train_ppl,  val_ppl))
    
        print('Model trained.')
        self.write_kaggle(model)
        print('Output saved.')
        
    def validate(self, model):
        criterion = nn.NLLLoss()
        aggregate_loss = []
        for batch in self.val_iter:
            
            x, y_t = self.batch_to_input(batch)
            y_p = model(x)
            
            loss = criterion(y_p, y_t)
            
            aggregate_loss.append(loss.data[0])
            
        val_ppl = np.exp(np.mean(aggregate_loss))
        return val_ppl
    
    def predict_sentence(self, string, model):
        string = string[:-4]
        x = self.string_to_batch(string)
        out_words = model.predict(x)
        return out_words
    
    def write_kaggle(self, model, input_file = 'input.txt'):
        inputs = open(input_file, 'r').read().splitlines()
        outputs = [self.predict_sentence(sentence, model) for sentence in inputs]
        with open('nnlm_output.txt', 'w') as f:
            f.write('id,word')
            for idx, line in enumerate(outputs):
                f.write('\n')
                f.write(str(idx) + ',')
                f.write(line) 

model = LanguageModel(hidden_dim = 1)
if use_cuda: 
    model.cuda()
trainer = Trainer(train_iter = train_iter, val_iter = val_iter)
trainer.train_model(model = model, num_epochs = 1)
