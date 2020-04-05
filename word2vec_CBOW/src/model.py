import torch
from torch import nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, word_2_id, window_size = 2, emb_dim = 5):
        super().__init__()
        self.C = window_size * 2
        self.word_2_id = word_2_id
        self.voc_size = len(self.word_2_id) 
        self.emb_dim = emb_dim
         
        self.U = nn.Embedding(self.voc_size, emb_dim)
        self.V = nn.Linear(emb_dim, self.voc_size)
        
        # weight initialization
        nn.init.kaiming_normal_(self.U.weight)
        nn.init.kaiming_normal_(self.V.weight)
        
    def forward(self, x):
        l1 = self.U.forward(x) 
        l1 = torch.sum(l1, dim=1) * (1/self.C)
        l2 = self.V.forward(l1) 
        return l2
    
    def get_vector(self, word):
        word_in_voc = self.word_2_id.get(word,0)
        uv_sum = self.U.weight[word_in_voc,:] + self.V.weight[word_in_voc,:] 
        return uv_sum/2


class CBOW_NS(nn.Module):
    def __init__(self, word_2_id, window_size = 2, emb_dim = 5):
        super().__init__()
        self.C = window_size * 2
        self.word_2_id = word_2_id
        self.voc_size = len(self.word_2_id) 
        self.emb_dim = emb_dim
         
        self.U = nn.Embedding(self.voc_size, emb_dim)
        self.V = nn.Linear(emb_dim, self.voc_size)
        
        # weight initialization
        nn.init.kaiming_normal_(self.U.weight)
        nn.init.kaiming_normal_(self.V.weight)
        
    def forward(self, x, target, neg_samples):
        l1 = self.U.forward(x) 
        
        h = (torch.sum(l1, dim=1) * (1/self.C)).unsqueeze(2)
        
        u_c = self.U(target).unsqueeze(1)
        u_k = self.U(neg_samples)
        
        pos_batch = F.logsigmoid(torch.bmm(u_c,h))
        pos = torch.sum(pos_batch)
        
        neg_batch = F.logsigmoid(torch.bmm(-u_k, h))
        neg = torch.sum(neg_batch)
        
        return -(pos + neg)
    
    def get_vector(self, word):
        word_in_voc = self.word_2_id.get(word,0)
        uv_sum = self.U.weight[word_in_voc,:] + self.V.weight[word_in_voc,:] 
        return uv_sum/2
