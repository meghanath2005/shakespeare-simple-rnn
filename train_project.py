# %% [markdown]
# Character-Level Text Prediction with a Simple RNN
# CSE 457 — Deep Learning Lab Project
#
# Tiny Shakespeare → character IDs → Embedding → simple tanh RNN → Linear → softmax.
# We predict the next character at every position in a 100-character window, giving
# many supervised examples per window. This is a vanilla RNN, not an LSTM or GRU.
# Training uses logits with CrossEntropyLoss (which includes log-softmax internally).
# Generation applies softmax explicitly. Run cells in order.
# %%
from pathlib import Path
import copy, hashlib, json, random, time, urllib.request
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

ROOT = Path.cwd()
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
print('PyTorch:', torch.__version__)
print('Training device:', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU')
CONFIG = dict(sequence_length=100, embedding_dim=64, hidden_size=128,
              batch_size=128, epochs=25, learning_rate=0.002, patience=5, seed=SEED)
print(CONFIG)
# %% [markdown]
# Dataset and leakage prevention
# Source: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
# Download once and reuse the local file. Split contiguous raw text 80/10/10 BEFORE
# constructing windows. Vocabulary is learned from training text only; an UNK
# token handles unseen validation/test characters. No windows cross split boundaries.
# %%
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
path = DATA / 'tiny_shakespeare.txt'
if not path.exists():
    urllib.request.urlretrieve(URL, path)
text = path.read_text(encoding='utf-8')
assert len(text) > 1_000_000, 'Dataset appears incomplete.'
a, b = int(len(text)*0.8), int(len(text)*0.9)
raw = {'train': text[:a], 'validation': text[a:b], 'test': text[b:]}
chars = sorted(set(raw['train']))
itos = ['<UNK>'] + chars
stoi = {c: i for i, c in enumerate(itos)}
encoded = {k: torch.tensor([stoi.get(c, 0) for c in v], dtype=torch.long)
           for k, v in raw.items()}
metadata = {'url': URL, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'characters': len(text), 'split_characters': {k: len(v) for k,v in raw.items()},
            'vocabulary': itos, 'config': CONFIG}
(OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Corpus characters:', len(text), '| vocabulary including UNK:', len(itos))
print(metadata['split_characters'])
print(text[:400])
# %%
class CharacterWindows(Dataset):
    def __init__(self, tokens, length):
        self.tokens, self.length = tokens, length
    def __len__(self):
        return (len(self.tokens)-1) // self.length
    def __getitem__(self, index):
        start = index * self.length
        return (self.tokens[start:start+self.length],
                self.tokens[start+1:start+self.length+1])

datasets = {k: CharacterWindows(v, CONFIG['sequence_length']) for k,v in encoded.items()}
loaders = {k: DataLoader(v, batch_size=CONFIG['batch_size'], shuffle=(k=='train'),
                        num_workers=0, pin_memory=(DEVICE.type=='cuda'))
           for k,v in datasets.items()}
print('Windows per split:', {k:len(v) for k,v in datasets.items()})
example_x, example_y = datasets['train'][0]
assert torch.equal(example_x[1:], example_y[:-1])
print('Input:', ''.join(itos[i] for i in example_x))
print('Target:', ''.join(itos[i] for i in example_y))
# %% [markdown]
# Model
# Hidden state is reset between independent windows. During generation we carry
# hidden state forward. Gradient clipping reduces exploding gradients. Validation
# loss selects the checkpoint; the test set is used only after model selection.
# %%
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True, nonlinearity='tanh')
        self.dense = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        states, hidden = self.rnn(self.embedding(x), hidden)
        return self.dense(states), hidden

model = CharRNN(len(itos), CONFIG['embedding_dim'], CONFIG['hidden_size']).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
print(model)
print('Trainable parameters:', sum(p.numel() for p in model.parameters()))

def run_epoch(loader, training=False):
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits,_ = model(x)
            loss = criterion(logits.reshape(-1,len(itos)), y.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item()*y.numel()
            correct += (logits.argmax(-1)==y).sum().item()
            count += y.numel()
    return loss_sum/count, correct/count
# %%
history = []
best_loss, stale = float('inf'), 0
started = time.time()
for epoch in range(1, CONFIG['epochs']+1):
    tick = time.time()
    train_loss, train_accuracy = run_epoch(loaders['train'], True)
    val_loss, val_accuracy = run_epoch(loaders['validation'])
    row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
               val_loss=val_loss, val_accuracy=val_accuracy, seconds=time.time()-tick)
    history.append(row)
    if val_loss < best_loss:
        best_loss, stale = val_loss, 0
        torch.save({'model_state': model.state_dict(), 'config': CONFIG,
                    'vocabulary': itos, 'epoch': epoch, 'val_loss': val_loss}, OUT/'best_model.pt')
    else:
        stale += 1
    (OUT/'history.json').write_text(json.dumps(history,indent=2), encoding='utf-8')
    message = (f'Epoch {epoch:02d}/{CONFIG["epochs"]} | loss {train_loss:.4f} | '
               f'acc {train_accuracy:.2%} | val loss {val_loss:.4f} | '
               f'val acc {val_accuracy:.2%} | {row["seconds"]:.1f}s')
    print(message, flush=True)
    with (OUT/'training.log').open('a', encoding='utf-8') as f:
        f.write(message+'\n')
    if stale >= CONFIG['patience']:
        print('Early stopping: validation loss stopped improving.')
        break
print(f'Training finished in {(time.time()-started)/60:.1f} minutes.')
# %% [markdown]
# Held-out evaluation
# Accuracy is the percentage of next characters correctly predicted under teacher
# forcing. It does not directly measure free-running generation quality. The baseline
# always predicts the most common training character, using the identical test targets.
# %%
checkpoint = torch.load(OUT/'best_model.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state'])
test_loss, test_accuracy = run_epoch(loaders['test'])
most_common = int(torch.bincount(encoded['train']).argmax())
baseline_correct = baseline_count = 0
for _,y in loaders['test']:
    baseline_correct += (y==most_common).sum().item()
    baseline_count += y.numel()
metrics = dict(best_epoch=checkpoint['epoch'], test_loss=test_loss,
               test_accuracy=test_accuracy, test_perplexity=float(np.exp(test_loss)),
               baseline_accuracy=baseline_correct/baseline_count,
               baseline_character=itos[most_common],
               device=torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU')
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2), encoding='utf-8')
print(json.dumps(metrics,indent=2))
fig, axes = plt.subplots(1,2,figsize=(12,4))
for key,label in [('train','Training'),('val','Validation')]:
    axes[0].plot([r['epoch'] for r in history],[r[key+'_loss'] for r in history],label=label)
    axes[1].plot([r['epoch'] for r in history],[r[key+'_accuracy'] for r in history],label=label)
for ax,title in zip(axes,['Cross-entropy loss','Next-character accuracy']):
    ax.set(xlabel='Epoch',title=title)
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT/'training_curves.png',dpi=150)
plt.show()
# %% [markdown]
# Generate text
# Temperature below 1 sharpens the distribution; above 1 increases randomness.
# Compare spelling, dialogue formatting, repetition, and coherence in the samples.
# A small simple RNN is expected to learn local patterns, not reliable story meaning.
# %%
@torch.inference_mode()
def generate(seed='ROMEO: ', length=400, temperature=0.8, random_seed=42):
    if not seed or temperature <= 0 or length < 0:
        raise ValueError('Use a nonempty seed, positive temperature, and nonnegative length.')
    unknown = set(seed)-set(stoi)
    if unknown:
        raise ValueError(f'Seed contains unknown characters: {unknown}')
    model.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(random_seed)
    x = torch.tensor([[stoi[c] for c in seed]],device=DEVICE)
    logits,hidden = model(x)
    result = seed
    for _ in range(length):
        scores = logits[0,-1].clone()/temperature
        scores[0] = -float('inf')  # never emit the UNK placeholder
        token = torch.multinomial(torch.softmax(scores,dim=-1),1,generator=rng)
        result += itos[token.item()]
        logits,hidden = model(token.reshape(1,1),hidden)
    return result

samples = []
for temperature in [0.5,0.8,1.2]:
    sample = f'--- Temperature {temperature} ---\n'+generate(temperature=temperature)
    samples.append(sample)
    print(sample+'\n')
(OUT/'generated_samples.txt').write_text('\n\n'.join(samples),encoding='utf-8')
# %%
# Change these values and rerun this cell for your live demonstration.
print(generate(seed='JULIET: ',length=500,temperature=0.8))
