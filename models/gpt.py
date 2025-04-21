import torch
import torch.nn as nn 
import torch.nn.functional as F 

from base import Block, DyT
from train import GPTConfig


class GPT(nn.Module):

  def __init__(self, config):
    super(GPT, self).__init__()
    assert config.vocab_size is not None
    assert config.n_embd is not None

    self.config = config

    # transformers blocks
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),        # word embeddings
        wpe = nn.Embedding(config.block_size, config.n_embd),        # position embeddings 
        drop = nn.Dropout(config.dropout),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = DyT(config) if config.DyT else nn.LayerNorm(config.n_embd, bias = config.bias)
    ))

    # finall prediction layer (we can call layer as output embedding matrix)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) 

    # weight tying (sharing) . by tying input embedding matrix and output embedding matrix (just logits predictor before softmax)
    # we can improve models performance and also it's reduce number of parameters used in our model 
    self.transformer.wte.weight = self.lm_head.weight

    # initializing model weights
    self.apply(self._init_weights)

    # report the parameters counts
    print("number of parameters: %.2fM" % (self.get_num_parameters()/ 1e6))

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean = 0.0, std = 0.2)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean = 0.0, std = 0.2)


  def get_num_parameters(self, non_embedding = True):
    """
    total number of paramters in our model
    for non_embedding we subtract positional embedding counts
    """
    total_params = sum(p.nelement() for p in self.parameters())
    if non_embedding:
      total_params -= self.transformer.wpe.weight.nelement()
    return total_params


  def forward(self, ids, targets = None):
    device = ids.device
    b, t = ids.shape   #(B, T)
    assert t <= self.config.block_size,  f"can't forward cause sequence length {t}, but block_size only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device = device) #(t,)

    wte = self.transformer.wte(ids)           # word embedding  (b, t, n_embd)
    wpe = self.transformer.wpe(pos)           # positional embedding  (t, n_embd)
    x = self.transformer.drop(wte + wpe)      # apply dropout before feed to blocks
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)              # normalization before prediction

    # if we have targets then we can calculate loss
    if targets is not None:
      logits = self.lm_head(x)                # (B, T, vocab_size)
      loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
    else:
      # duing infernce we only pass last token to predict next token eventhough we read all token before that
      logits = self.lm_head(x[:, [-1], :])
      loss = None

    return logits, loss
  
  @classmethod
  def from_pretrained(cls, model_type, override_args = None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

    override_args = override_args or {}                # we can additionaly override dropout only
    assert all(k == 'dropout' for k in override_args)

    print(f"loading weights from pretrained gpt : {model_type}")
    from transformers import GPT2LMHeadModel

    # gpt2 models 
    config_args = {
              'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
              'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
              'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
              'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params or ~ 1.5B
          }[model_type]


    print('forcing vocab_size 50257, block_size 1024, bias = True')

    config_args['vocab_size'] = 50257      # gpt2 has always vocab size of 50257
    config_args['block_size'] = 1024
    config_args['bias'] = True
    # optionally we can override dropout rate, if needed
    if 'dropuout' in override_args:
      print(f"overriding dropout rate to {override_args['dropout']}")
      config_args['dropout'] = override_args['dropout']

    # initializing model from scartch with new config 
    config = GPTConfig(**config_args)
    model = GPT(config) 
    sd = model.state_dict() 
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]        # no need to include masking

    # pretrained gpt2 model state dict (optionally if we have locally downloaded weights we can use that without downloading again)
    m_path = "./gpt2-local" if "./gpt2-local" else model_type
    model_hf = GPT2LMHeadModel.from_pretrained(m_path)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = sd_hf.keys()
    # these are the masking so we don't want that 
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked.bias')]

    transposed =  ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # so in order to copy trained gpt2's weights to our model we have change shape of the layer that they used cause
    # as default they use conv1x1 layer as projection paramters here we are using linear layer so we have just make sure
    # thier shape matches after all conv1x1 just a like normal feed forward operation

    assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys : {len(sd_keys_hf) != len(sd_keys)}"

    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # here we doing transpose of conv1x1 weights so it can match shpae of our linear layer
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t()) 
      
      else:
        # just normal copy of parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k]) 

    print(f"pretrained {model_type} weights are loaded successfully!")

    return model

  def generate(self, idx, max_tokens, temperature =1.0):
    # idx (b, t) by takes previous sequence we try to complete the sequence so
    # every iteration we increase t size
    with torch.no_grad():
      for _ in range(max_tokens):
        # we croping the block size we can take infinite pre-context to predict next token
        idx_count = idx if idx.size(1) <= self.config.block_size else idx[:,-self.config.block_size:]
        # get the logits from model
        logits, _ = self(idx_count)
        # then scale the logits by temperature. by doing this way we can control how next token going to draw
        logits = logits[:, -1, :] / temperature
        # then apply softmax to get prob distripution for our vocab
        probs = torch.softmax(logits, dim = -1)
        # we drawing next token in random sampling way so even token with lowest prob will get a chance
        next_idx = torch.multinomial(probs, num_samples=1)
        # then add the next token to our token seq so next time model can predict token based on this token
        idx = torch.cat((idx, next_idx), dim=1)

    return idx
  

if __name__ == '__main__':
  gpt = GPT.from_pretrained('gpt2').to(device)
  print(gpt)