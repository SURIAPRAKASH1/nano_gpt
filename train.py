import torch
import torch.optim as optim
from models.gpt import GPT
import tiktoken

from dataclasses import dataclass
import math
from collections import defaultdict


# current device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device', device) 

# open the text file that contains EM's lyrics
eminem_text = open('All_eminem_songs.txt').read()
text = eminem_text

# we are using gpt-4 tokenizer it has vocab size of 100277 . 
# so our text going to get encoding in tokens range of (0 - 100276)
# so in gpt2 if we are going to use pretrained gpt2 then we should better use gpt2 tokenizer 
is_pretrained_model = True if input("pretrained model y/n: ") == 'y' else False
tokenizer =  tiktoken.get_encoding('gpt2')
tokens = tokenizer.encode(text)                     # we don't use any special tokens
vocab_size = tokenizer.n_vocab

print("vocab size", vocab_size)
print(f"total number of charactors in our text is {len(text)} get tokenized into {len(tokens)} tokens")

# gpt model hyperparameters 
@dataclass
class GPTConfig:
    n_embd: int = 128                # just vector representation dim for each token in sequence (block)
    block_size: int = 16           # how many tokens in one block ?
    batch_size: int = 32             # how many blocks as group ?
    n_head:int = 6                  # number of self attention in paralell (actually scaled dot product attention)
    vocab_size: int = vocab_size         # all posible unique tokens 
    DyT: bool = False             
    n_layer: int = 4
    dropout: float = 0.2
    bias: bool = True
    alpha: float = 0.5 

# optimizer's hyperparameters
weight_decay = 1e-1    # in adaptive gradinet optimizers like adam weight_decay != L2_regularization 
lr = 6e-4              # initial learning rate
betas = (0.9, 0.95)    # momentums
min_lr = 6e-5          
num_steps = 5000       # total number of steps used to train model
eval_interval = 500    # for calculation loss once in while
eval_iters = 200       # how many batches we should take to compute model loss
warmup_iters = 200     # we wanna use small lr during initial 


torch.cuda.manual_seed(1337) if torch.cuda.is_available() else torch.manual_seed(1337) # for reproducibility
# train and dev dataset splits (so if we want we can split our data as three splits (train, dev, test))
n = int(len(tokens) * 0.9)
train_data = tokens[:n]         # 90% train data
dev_data =  tokens[n:]          # 10% dev data
print(f"train dataset tokens: {len(train_data)}\ndev dataset tokens {len(dev_data)}")


# get batch of examples
def get_batch(split, device):

    data = train_data if split == 'train' else dev_data
    xi = torch.randint(len(data) - GPTConfig.block_size, (GPTConfig.batch_size, ))
    x = torch.tensor([data[i: i + GPTConfig.block_size] for i in xi])
    y = torch.tensor([data[i + 1: i + GPTConfig.block_size + 1] for i in xi])
    
    # ----------------------  only for gpu 🚀----------------------
    # as default pytorch tensor is pageable (means it lives in virtual memory (ram and disk))
    # so when we wanna send our tensor to gpu (so we can execute there) . if a tensor is pageable then it must come to ram
    # then from ram it can send to gpu . but it takes time like (pageable_memory --> ram --> gpu)
    # so by pinning our tensor like x.pin_memory() , we make tensor only live in (pinned_memory) ram 
    # becuase of the we can efficiently transfar our tensor to gpu
    # what about non_blocking ? . so when one batch is transfared to gpu we don't want to wait to 
    # transfaring anothor batch (async) . so this way gpu never get's idel so we use gpu in it's full power
    
    if device != 'cpu':
        x = x.pin_memory()
        y = y.pin_memory()
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)
    else:
        # else just do your normal thing bro ...
        x = x.to(device)
        y = y.to(device)

    return x, y


def get_lr(it):
  
  # so we gradually increasing learning rate 
  if it < warmup_iters:
    return  lr * (it + 1) / (warmup_iters + 1)
  
  # starting to decaying the learning rate using cosine 
  else:
    decay_ratio = (it - warmup_iters) / (num_steps - warmup_iters)
    assert 0 <= decay_ratio <=1 
    coeff = 0.5 * ( 1.0 + math.cos( math.pi * decay_ratio))
    return  min_lr + coeff * (lr - min_lr)    # we make sure learning rate shouldn't 0 (but we wanna decrease) 


# for estimating loss over some number of batches instead of per-batch (so noise) or per-epoch (we can do that)
@torch.no_grad()
def estimate_loss():
  model.eval()

  out = {}
  for split in ['train', 'dev']:
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
      x, y = get_batch(split, device)
      _, loss = model(x, y)
      losses[i] = loss.item()

    # take means over batches
    averge_loss = losses.mean()
    out[split] = averge_loss

  model.train()
  return out

# if we wanna see how transformer going to look at each token (in generative way) just call this function
def transfomers_view():
    X, Y = get_batch('train', device)
    # how model going to see and predict next token
    for i in range(GPTConfig.block_size):
        x = X[7][:i+1].tolist()
        y = Y[7][i].tolist()
        print(x,"-->", y,"\n",tokenizer.decode(x), "-->", tokenizer.decode([y]))

# the loss curves ☹
def plot_loss_curve(gb_lossi):
    # train and dev loss curve
    plt.plot(gb_lossi['train'])
    plt.plot(gb_lossi['dev'])
    plt.ylabel('Loss')
    plt.xlabel(f'Per {eval_iters} Batches')

    plt.legend(['train', 'dev'])
    plt.show()

def model_train(model):
    # optimizer
    optimizer = optim.AdamW(model.parameters(), betas= betas, weight_decay= weight_decay)
   
    # optimization loop
    pb_lossi = []                   # loss for per-batch
    gb_lossi = defaultdict(list)    # loss for some bunch of batches

    for step in range(num_steps):
        # get batch of (x, y) pair
        X, Y = get_batch('train', device)

        # evaluate model once on while
        if step % eval_interval == 0 or step  == num_steps - 1:
            out = estimate_loss()
            # for plotting loss curve
            gb_lossi['train'].append(out['train'].item())
            gb_lossi['dev'].append(out['dev'].item())

            print(f"step {step}/{num_steps}: train_loss {out['train']}, dev_loss {out['dev']}")

        # forward pass and compute loss
        logits, loss = model(X, Y)
        optimizer.zero_grad()
        # back-ward pass
        loss.backward()
        # update parameters
        optimizer.defaults['lr'] = get_lr(step)
        optimizer.step()
        pb_lossi.append(loss.item())

    plot_loss_curve(gb_lossi)
    return model


# so if we use pretrained weights then we should mention gpt2's size
model = GPT(GPTConfig).from_pretrained('gpt2') if is_pretrained_model else GPT(GPTConfig).to(device)

# trian model if not pretrained 
trained_model = model if is_pretrained_model else model_train(model)

# generate bars 
query = """look """
max_tokens = 5
trained_model.eval()
# sampling from model
encoded_query = torch.tensor([tokenizer.encode(query)], device = device)
result = tokenizer.decode(trained_model.generate(encoded_query, temperature= 0.8, max_tokens = max_tokens)[0].tolist())
print(result)
