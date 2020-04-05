from collections import Counter
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader, Dataset


class Batcher(Dataset):    
    def __init__(self, text_file, window_size=4, voc_threshold = 1000):
        self.tokens = text_file.readline().split()[:]
        self.word_2_id, self.id_2_word = self.build_dicts(num_first = voc_threshold)
        
        self.tokens = [self.word_2_id.get(w, 0) for w in self.tokens]
        
        self.voc_size = len(self.word_2_id)
        self.len_t = len(self.tokens)
        self.window_size = window_size
        
    def __getitem__(self, idx):
        
        l = max(0, idx - self.window_size)
        r = min(self.len_t, idx+1+self.window_size) 
        context = self.tokens[l:idx] + self.tokens[idx+1:r]
        
        to_pad = (self.window_size * 2) - len(context)
        context.extend([self.UNK_ID] * to_pad)        
        
        return  tensor(context), self.tokens[idx]
        
    def __len__(self):
        return len(self.tokens)
    
    def build_dicts(self, num_first):
        words_for_dict =[w for w,f in Counter(self.tokens).most_common(num_first)]
        vocabulary = {k:idx for idx, k in enumerate(words_for_dict, 1)}
        word_2_id = {**{"<UNK>":0}, **vocabulary}
        self.UNK_ID = word_2_id["<UNK>"]
        id_2_word = {v:k for k,v in word_2_id.items()}
        return word_2_id, id_2_word
    
    
    def indices_to_words(self, x_batch):
        return np.vectorize(self.word_2_id.get)(x_batch)
    

class BatcherNS(Dataset): 
    def __init__(self, text_file, neg_samples, window_size=4, voc_threshold = 1000):
        self.tokens = text_file.readline().split()[:]
        self.word_2_id, self.id_2_word = self.build_dicts(num_first = voc_threshold)
        
        self.tokens = [self.word_2_id.get(w, 0) for w in self.tokens]
        
        self.voc_size = len(self.word_2_id)
        self.len_t = len(self.tokens)
        self.window_size = window_size
        
        
        self.neg_samples = 10
        self.unigram_dist = self.build_unigram_distribution()
    
    def __getitem__(self, idx):
        
        l = max(0, idx - self.window_size)
        r = min(self.len_t, idx+1+self.window_size) 
        context = self.tokens[l:idx] + self.tokens[idx+1:r]
        
        to_pad = (self.window_size * 2) - len(context)
        context.extend([self.UNK_ID] * to_pad)        
        
        neg_samples = np.random.choice(range(self.voc_size), size=self.neg_samples, p = self.unigram_dist)
        
        return  tensor(context), self.tokens[idx], tensor(neg_samples)

    def __len__(self):
        return len(self.tokens)
    
    def build_dicts(self, num_first):
        dict_with_freq = Counter(self.tokens).most_common(num_first)
        words_for_dict = [w for w,f in dict_with_freq]
        self.words_freq = [f for w,f in dict_with_freq]
        
        vocabulary = {k:idx for idx, k in enumerate(words_for_dict, 1)}
        word_2_id = {**{"<UNK>":0}, **vocabulary}
        
        self.UNK_ID = word_2_id["<UNK>"]
        
        id_2_word = {v:k for k,v in word_2_id.items()}
        return word_2_id, id_2_word
    
    
    def indices_to_words(self, x_batch):
        return np.vectorize(self.word_2_id.get)(x_batch)
    
            
    def build_unigram_distribution(self): 
        unk_freq = self.len_t - sum(self.words_freq)

        u_distr = np.concatenate([[unk_freq],self.words_freq]).astype("float")        
        u_distr /= self.len_t # convert to probabilities
        
        u_distr = np.power(u_distr,3/4) # increase probability of "scarse" words
        u_distr /= sum(u_distr)
        
        assert abs(sum(u_distr) - 1) < 1e-6 # check whether sum up to 1
        
        return u_distr