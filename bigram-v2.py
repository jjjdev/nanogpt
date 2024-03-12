import torch
import torch.nn as nn

from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # Number of independent sequences to process in parallel
block_size = 8 # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 0.001
eval_iters = 200
n_embd = 32 # number of embedding dimensions

# Set the device to use cuda if there's a GPU.  
# Note that later we move the model and data to the device (by passing device in args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)

# First lets get the tiny shakespeare dataset - uncomment to download again!
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Load the data
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Find the unique characters that occur in the text
chars = sorted(list(set(text)))
vocabulary_size = len(chars)

# Create very simple tokenizer - maps characters to integers and vice versa
# Look at the other tokenizers that are used.
# Google uses SentencePiece.  OpenAI uses tiktoken.
str_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_str = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [str_to_int[ch] for ch in s] # encoder - converts string to integers
decode = lambda x: ''.join([int_to_str[i] for i in x]) # decoder - converts integers to string

# Encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Create train and test data sets
# We will use 90% of the data for training and 10% for validation (change n for different percentages)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Disables Backward Propagation for more efficient pytorch memory use
def estimate_loss():
    out = {}
    model.eval() # Set model for evaluation phase
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item() # get loss
        out[split] = losses.mean() # get avg loss over both splits (a lot less noisy)
    model.train() # Set model back to training phase
    return out

# Simple Bigram Model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Each token directly reads off the logits for the next token from a lookup table
        # has a weight inside that stores the probability of the next token
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # each pos gets own embedding vector 
        self.linear = nn.Linear(n_embd, vocabulary_size)

        # I left these there but they aren't needed
        #self.embedding = nn.Embedding(vocabulary_size, vocabulary_size)
        #self.linear = nn.Linear(vocabulary_size, vocabulary_size)

    def forward(self, idx, targets=None):

        # idx and targers are both (B,T) tensor of integers
        #logits = self.token_embedding_table(idx) # (B, T, C)
        tok_emb = self.token_embedding_table(idx) # (B, T, C) # instead of tokens get token embeddings
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))    # (T, C)
        # To go from token embeddings to logits, need a linear layer
        logits = self.linear(tok_emb) # (B, T, C) c = vocabulary size

        if targets is None:
            loss = None
        else:
            # reshape logits so we can use cross-entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)  # -1 means "infer this dimension" (translates to B*T)

            # evaluate the loss function (quality of predictions)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
    # idx is a (B, T) array of indices in the current 
    # the job of generate is to take a BxT and return a BxT+1, +2, +3, etc

        for _ in range(max_new_tokens):
            
            # Get the predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)    # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
# Create the model
model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # Every once in a while evaluate the loss on the training and validation data
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter} - Train Loss: {losses["train"]:.4f}, Validation Loss: {losses["validation"]:.4f}')

    # Get a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device) # creating a 1x1 tensor of zeros (remember 0 = new line)

# ask for 500 new tokens, generate, convert to list to feed into decode
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))