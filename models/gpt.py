import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from dataclasses import dataclass
from base import Block, DyT

@dataclass
class GPTConfig:
    n_embd: int = 96
    block_size: int = 8
    batch_size: int = 32
    n_head:int = 4
    vocab_size = 10
    n_layers = 2
    dropout: float = 0.2
    bias: bool = False
    alpha: float = 0.5 


class GPT(nn.Module):

    def __init__(self, config: GPTConfig ):
        super(GPT, self).__init__()

        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.transformer = nn.ModuleDict(dict(
          wte = nn.Embedding(config.vocab_size, config.n_embd),
          pte = nn.Embedding(config.block_size, config.n_embd),
          h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]) ,
          fcl = nn.Linear(config.n_embd, config.n_embd),
          dyt = DyT(GPTConfig),
        ))
        

        # finall prediction layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, ids, targets = None):
        device = ids.device
        b, t = ids.shape
        assert t == self.block_size,  f"can't forward with block size  {t}, expected block size {self.block_size}"
        pos = torch.arange(0, t, dtype= torch.long, device = device)     # (t, )

        # token embeddings and positional embedding for given ids (just n dim learnable vector)
        wte = self.transformer.wte(ids)             # (b, t, n_embd)
        pte  = self.transformer.pte(pos)            # (t, n_embd)
        
        # normalization before feeding to transfomer
        x = self.transformer.dyt( wte + pte )
        for block in self.transformer.h:
            x = block(x) 
        # finnal projection and normalization
        x = self.transformer.dyt(self.transformer.fcl(x))

        if targets is not None:
            logits = self.lm_head(x) 
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        else:
            # here little optimaization: during inference we only take last token 
            # to predict next token . but our transfomer reads whole sequence only last prediction layer just modified
            logits = self.lm_head(x[:, [-1], :])
            loss = None 
        
        return logits, loss









