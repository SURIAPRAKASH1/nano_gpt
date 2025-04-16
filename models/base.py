import torch.nn as nn
import torch
from gpt import GPTConfig


# Dynamic Tanh as normalizer
class DyT(nn.Module):

    def __init__(self, config):
        super(DyT, self).__init__()
    
        # alpha for scaling input
        self.alpha = nn.Parameter(torch.ones(1) * config.alpha) 
        # scale and shift
        self.gamma = nn.Parameter(torch.ones(config.n_embd)) 
        self.beta = nn.Parameter(torch.zeros(config.n_embd)) 

    def forward(self, x):
        x = torch.tanh( x * self.alpha ) 
        return self.gamma * x + self.beta


# MultiHead Attention Layer
class MultiHeadAttention(nn.Module):
    "MultiHeadAttention just a for loop over Scaled dot-product Attention  A(q, k, v)= (softmax(QK^t) / sqrt(dk)) V "

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config.n_embd % config.n_head == 0 

        self.n_embd = config.n_embd
        self.n_head = config.n_head 
        self.block_size = config.block_size

        # projection parameters for to generate q, k, v in batch level
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias= config.bias) 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias) 

        # regularization (dropout)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout) 

        # masking 
        self.register_buffer('tril', torch.tril(
            torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)
        ))

    def forward(self, x):
        B, T, C = x.shape             # (B, T, C) --> (batch_size, block_size, n_embd) 
        
        # q, k, v for each token in batch level and move the head dim towards to batch diem
        q, k, v = self.attn(x).split(split_size = self.n_embd, dim = -1)   
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, T, n_head, hs) -> (B, n_head, T, hs) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, T, n_head, hs) -> (B, n_head, T, hs) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, T, n_head, hs) -> (B, n_head, T, hs) 

        # attention scores and we scaling that so the current token will atten to all past token to get self attention
        # else current token don't attent to past tokens so it can't fully understand what's going on it's arround that
        # but actually token going get attentions from left to right cause of sequence 
        attn = q @ k.transpose(-1, -2) / (q.size(-1) **0.5)       # (B, n_head, T, hs) @ (B, n_head, hs, T) --> (B, n_head, T, T) 
        # masking out right tokens so model can only predict next token by leanring past tokens only
        attn = attn.masked_fill_(self.tril[:, :, :T, :T] == 0, float('-inf'))
        # attention weights
        attn = torch.softmax(attn , dim = -1) 
        attn = self.attn_dropout(attn)
        # aggreate value with attention weights
        out = attn @ v                          # (B, n_head, T, T) @ (B, n_head, T, hs ) --> (B, n_head, T, hs) 

        # by collapsing n_head and hs dim we get cancat head attention and by projecting that we get multihead attention  
        out = out.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C) 
        out = self.residual_dropout(self.c_proj(out))      

        return out 


# MLP Layer
class MLP(nn.Module):

    "It's fully connected layers for doing upward and downward projection to ouput of Attention layer"

    def __init__(self, config):
        super(MLP, self).__init__()

        # here we have upward and downward projection parameters
        self.fcl = nn.Linear(config.n_embd, 4* config.n_embd)
        self.act = nn.GELU() 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.fcl(x)
        x = self.act(x)
        x = self.c_proj(x) 
        return x 
    

class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__() 

        # multihead attention layer
        self.m_attn = MultiHeadAttention(config)
        # dense layer
        self.mlp = MLP(config) 
        # dynamic tanh normalization
        self.dyt = DyT(config) 
    
    def forward(self, x):
        x = x + self.m_attn(self.dyt(x))
        x = x + self.mlp(self.dyt(x))
        return x 
    



if __name__ == '__main__':
    b = Block(GPTConfig)

    x = torch.randn((GPTConfig.batch_size, GPTConfig.block_size, GPTConfig.n_embd))

    print(b(x).shape, b)
