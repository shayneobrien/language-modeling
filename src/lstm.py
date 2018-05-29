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

class LSTMLanguageModel(nn.Module):
    """ simple LSTM neural network language model """     
    def __init__(self, hidden_dim = 100, TEXT = TEXT, batch_size = 10):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        vocab_size, embedding_dim = TEXT.vocab.vectors.shape
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.hidden_dim, dropout = 0.50)
        self.linear = nn.Linear(in_features = self.hidden_dim, out_features = vocab_size)
        self.drop = nn.Dropout(p = 0.50)

    def init_hidden(self):
        direction = 2 if self.lstm.bidirectional else 1
        if use_cuda:
            return (Variable(torch.zeros(direction*self.lstm.num_layers, self.batch_size, self.hidden_dim)).cuda(), 
                    Variable(torch.zeros(direction*self.lstm.num_layers, self.batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(direction*self.lstm.num_layers, self.batch_size, self.hidden_dim)), 
                    Variable(torch.zeros(direction*self.lstm.num_layers, self.batch_size, self.hidden_dim)))
    
    def detach_hidden(self, hidden):
        """ util function to keep down number of graphs """
        return tuple([h.detach() for h in hidden])
        
    def forward(self, x, hidden, train = True):
        """ predict, return hidden state so it can be used to intialize the next hidden state """
        embedded = self.embeddings(x)
        embedded = self.drop(embedded) if train else embedded
        
        lstm_output, hdn = self.lstm(embedded, hidden)
        reshaped = lstm_output.view(-1, lstm_output.size(2))
        dropped = self.drop(reshaped) if train else reshaped
        
        decoded = self.linear(dropped)
        
        logits = F.log_softmax(decoded, dim = 1)
                
        return logits, self.detach_hidden(hdn)    
    
class Trainer:
    def __init__(self, train_iter, val_iter):
        self.train_iter = train_iter
        self.val_iter = val_iter
        
    def string_to_batch(self, string):
        relevant_split = string.split() # last two words, ignore ___
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
        
        for epoch in tqdm_notebook(range(num_epochs)):

            epoch_loss = []
            hidden = model.init_hidden()
            model.train()

            for batch in tqdm_notebook(train_iter):
                x, y = batch.text, batch.target.view(-1)
                if use_cuda: x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()

                y_pred, hidden = model.forward(x, hidden, train = True)

                loss = criterion(y_pred, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm(model.lstm.parameters(), 1)

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
        hidden = model.init_hidden()
        aggregate_loss = []
        for batch in val_iter:
            y_p, _ = model.forward(batch.text, hidden, train = False)
            y_t = batch.target.view(-1)
            
            loss = criterion(y_p, y_t)
            aggregate_loss.append(loss.data[0])        
        val_ppl = np.exp(np.mean(aggregate_loss))
        return val_ppl
    
    def predict_sentence(self, string, model, TEXT = TEXT):
        string = string[:-4]
        model.batch_size = 1
        hidden = model.init_hidden()
        x = self.string_to_batch(string)
        logits, _ = model.forward(x, hidden, train = False)
        argsort_ids = np.argsort(logits[-1].data.tolist())
        out_ids = argsort_ids[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words
    
    def write_kaggle(self, model, input_file = 'input.txt'):        
        inputs = open(input_file, 'r').read().splitlines()
        outputs = [self.predict_sentence(sentence, model) for sentence in inputs]
        with open('lstm_output.txt', 'w') as f:
            f.write('id,word')
            for idx, line in enumerate(outputs):
                f.write('\n')
                f.write(str(idx) + ',')
                f.write(line) 

model = LSTMLanguageModel(hidden_dim = 1024)
if use_cuda: 
    model.cuda()
trainer = Trainer(train_iter = train_iter, val_iter = val_iter)
trainer.train_model(model = model, num_epochs = 10)

