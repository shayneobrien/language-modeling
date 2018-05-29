import torchtext, random, torch, argparse

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
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test), batch_size=128, device=device, bptt_len=32, repeat=False)

class LSTMLanguageModel(nn.Module):
    def __init__(self, hidden_dim, num_layers = 2, tieweights = True, init_params = False, TEXT = TEXT):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        vocab_size, embedding_dim = TEXT.vocab.vectors.shape
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.hidden_dim, num_layers=num_layers, dropout = 0.50)
        self.drop = nn.Dropout(p = 0.50)
        if tieweights:
            self.linear = nn.ModuleList([nn.Linear(in_features = self.hidden_dim, out_features = embedding_dim),
                                        nn.Linear(in_features = embedding_dim, out_features = vocab_size)])
            self.linear[1].weight = self.embeddings.weight
        else:
            self.linear = nn.Linear(in_features = self.hidden_dim, out_features = vocab_size)
        
        if init_params:
            self.init_lstm_params_uniformly(bound = 0.04)
        
    def init_lstm_params_uniformly(self, bound):
        for layer_params in self.lstm._all_weights:
            for param in layer_params:
                if 'weight' in param:
                    nn.init.uniform(self.lstm.__getattr__(param), -bound, bound)
                    
                    
    def init_hidden(self, batch_size):
        direction = 2 if self.lstm.bidirectional else 1
        if use_cuda:
            return (Variable(torch.zeros(direction*self.lstm.num_layers, batch_size, self.hidden_dim)).cuda(), 
                    Variable(torch.zeros(direction*self.lstm.num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(direction*self.lstm.num_layers, batch_size, self.hidden_dim)), 
                    Variable(torch.zeros(direction*self.lstm.num_layers, batch_size, self.hidden_dim)))
    
    def detach_hidden(self, hidden):
        """ util function to keep down number of graphs """
        return tuple([h.detach() for h in hidden])
        
    def forward(self, x, hidden, train = True):
        """ predict, return hidden state so it can be used to intialize the next hidden state """
        embedded = self.embeddings(x)
        # embedded = self.drop(embedded) if train else embedded
        
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
        self.batch_size = self.train_iter.batch_size
        
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
    
    def train_model(self, model, num_epochs, lr = 1e-2, decay = 1e-4, betas = (0.9, 0.999), clip = 2):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr=lr, weight_decay=decay betas = betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, patience = 1, factor = .25, threshold = 1e-3)
        
        weight = torch.FloatTensor(len(TEXT.vocab.itos)).fill_(1)
        padding_id = TEXT.vocab.stoi["<pad>"]
        weight[padding_id] = 0
        if use_cuda:
            weight = weight.cuda()
        
        criterion = nn.CrossEntropyLoss(weight = Variable(weight), size_average = False)
        
        for epoch in tqdm(range(num_epochs)):

            train_loss, train_words = 0, 0
            hidden = model.init_hidden(self.batch_size)
            model.train()

            for batch in tqdm(train_iter):
                x, y = batch.text, batch.target.view(-1)
                if use_cuda: x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()

                y_pred, hidden = model.forward(x, hidden, train = True)

                batch_loss = criterion(y_pred, y)
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm(model.lstm.parameters(), clip)

                optimizer.step()

                train_loss += batch_loss.data[0]
                train_words += y.ne(padding_id).int().sum().data[0]
                
            epoch_loss = train_loss / train_words
            train_ppl = np.exp(epoch_loss)
            model.eval()
                        
            val_loss, val_ppl = self.validate(model, criterion, padding_id)
            # dynamic LR optimization
            scheduler.step(val_loss)

            print('Epoch {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch+1, epoch_loss, train_ppl,  val_ppl))
    
        print('Model trained.')
        self.write_kaggle(model)
        print('Output saved.')
        torch.save(model.cpu(), model.__class__.__name__ + ".pth")
        
    def validate(self, model, criterion, padding_id):
        hidden = model.init_hidden(batch_size = self.val_iter.batch_size)
        aggregate_loss, aggregate_words = 0, 0
        for batch in self.val_iter:
            y_p, _ = model.forward(batch.text, hidden, train = False)
            y_t = batch.target.view(-1)
            
            batch_loss = criterion(y_p, y_t)
            
            aggregate_loss += batch_loss.data[0]
            aggregate_words += y_t.ne(padding_id).int().sum().data[0]
        
        val_loss = aggregate_loss / aggregate_words
        val_ppl = np.exp(val_loss)
        return val_loss, val_ppl
    
    def predict_sentence(self, string, model, TEXT = TEXT):
        string = string[:-4]
        hidden = model.init_hidden(batch_size = 1)
        x = self.string_to_batch(string)
        logits, _ = model.forward(x, hidden, train = False)
        argsort_ids = np.argsort(logits[-1].data.tolist())
        out_ids = argsort_ids[-20:][::-1]
        out_words = ' '.join([TEXT.vocab.itos[out_id] for out_id in out_ids])
        return out_words
    
    def write_kaggle(self, model, input_file = 'input.txt'):        
        inputs = open(input_file, 'r').read().splitlines()
        outputs = [self.predict_sentence(sentence, model) for sentence in inputs]
        with open('lstm_initialized_output.txt', 'w') as f:
            f.write('id,word')
            for idx, line in enumerate(outputs):
                f.write('\n')
                f.write(str(idx) + ',')
                f.write(line) 

model = LSTMLanguageModel(hidden_dim = 512, num_layers = 2)
if use_cuda: 
    model.cuda()
trainer = Trainer(train_iter = train_iter, val_iter = val_iter)
trainer.train_model(model = model, num_epochs = 50)

