# %% [markdown]
# Extended visual analysis
# These plots use the saved best checkpoint and held-out test data, without retraining.
# Test diagnostics are descriptive, not used to select the model. Generated-text
# statistics are proxies for style and diversity, not measures of meaning or literary quality.
# %%
from pathlib import Path
import json, math
from collections import Counter
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path.cwd()
OUT = ROOT/'outputs'
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
saved = torch.load(OUT/'best_model.pt', map_location=device, weights_only=True)
cfg, vocab = saved['config'], saved['vocabulary']
lookup = {c:i for i,c in enumerate(vocab)}
class AnalysisRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab),cfg['embedding_dim'])
        self.rnn = nn.RNN(cfg['embedding_dim'],cfg['hidden_size'],batch_first=True)
        self.dense = nn.Linear(cfg['hidden_size'],len(vocab))
    def forward(self,x,h=None):
        z,h = self.rnn(self.embedding(x),h)
        return self.dense(z),h
net = AnalysisRNN().to(device)
net.load_state_dict(saved['model_state'])
net.eval()
corpus = (ROOT/'data/tiny_shakespeare.txt').read_text(encoding='utf-8')
a,b = int(len(corpus)*.8),int(len(corpus)*.9)
parts = {'Train':corpus[:a],'Validation':corpus[a:b],'Test':corpus[b:]}
hist = json.loads((OUT/'history.json').read_text())
metrics = json.loads((OUT/'metrics.json').read_text())
def label(c):
    return {' ':'space','\n':'newline','\t':'tab'}.get(c,c)
def finish(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.show()
    plt.close(fig)
print('Loaded best checkpoint from epoch',saved['epoch'],'on',device)
# %% [markdown]
# 1. Project flowchart and model architecture
# Training and validation select the checkpoint. Held-out evaluation and generation
# use that selected checkpoint. Softmax is implicit in cross-entropy during training.
# %%
fig,ax = plt.subplots(figsize=(14,10))
ax.set(xlim=(0,14),ylim=(0,11))
ax.axis('off')
def box(x,y,w,h,text,color='#e8f0fe'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#355b85'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(start,end,text=None):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.7,color='#355b85'))
    if text:
        ax.text((start[0]+end[0])/2+.12,(start[1]+end[1])/2,text,fontsize=9)
box(4.5,9.7,5,0.8,'Tiny Shakespeare text corpus')
box(4.5,8.2,5,0.9,'Split raw text: 80% / 10% / 10%\nBuild vocabulary from training text')
arrow((7,9.7),(7,9.1))
box(.5,6.4,3.5,.9,'Training windows\n100 input → 100 shifted targets')
box(5.25,6.4,3.5,.9,'Validation windows\nNo overlap across splits')
box(10,6.4,3.5,.9,'Held-out test windows\nUsed after model selection')
for x in [2.25,7,11.75]: arrow((7,8.2),(x,7.3))
box(.5,4.4,8.25,1.1,'Character IDs → Embedding (64) → Simple tanh RNN (128)\n→ Dense (vocabulary logits) → next-character probabilities')
arrow((2.25,6.4),(2.25,5.5)); arrow((7,6.4),(7,5.5))
box(.5,2.4,3.5,1,'Adam + cross-entropy\nGradient clipping\nUp to 25 epochs', '#fff1d6')
box(5.25,2.4,3.5,1,'Select lowest validation loss\nSave best model\nEarly stopping patience: 5','#fff1d6')
arrow((2.25,4.4),(2.25,3.4)); arrow((4,2.9),(5.25,2.9))
arrow((7,4.4),(7,3.4))
box(10,2.4,3.5,1,'Test evaluation\nAccuracy, loss, perplexity\nError and confidence analysis','#e2f3e6')
arrow((11.75,6.4),(11.75,3.4)); arrow((8.75,2.9),(10,2.9))
box(4.5,.4,5,1,'Seed text → softmax(logits / temperature)\nSample character → feed back → repeat','#e2f3e6')
arrow((7,2.4),(7,1.4))
ax.set_title('Character-level text prediction: complete project workflow',fontsize=17,pad=14)
finish(fig,'01_project_flowchart')
# %% [markdown]
# 2. Dataset composition and character distribution
# Character frequencies include all raw characters in each split. Evaluation below
# uses complete 100-character windows and drops the short remainder at each split end.
# %%
fig,axes=plt.subplots(1,2,figsize=(14,4))
axes[0].bar(parts.keys(),[len(t) for t in parts.values()],color=['#3267a8','#ec9d35','#419e75'])
axes[0].set(title='Contiguous dataset split',ylabel='Characters')
for i,t in enumerate(parts.values()): axes[0].text(i,len(t),f'{len(t):,}',ha='center',va='bottom')
common=[c for c,_ in Counter(parts['Train']).most_common(20)]
for i,(name,t) in enumerate(parts.items()):
    counts=Counter(t)
    axes[1].bar(np.arange(20)+(i-1)*.26,[100*counts[c]/len(t) for c in common],.26,label=name)
axes[1].set(xticks=np.arange(20),xticklabels=[label(c) for c in common],ylabel='Frequency (%)',title='Top 20 training characters across splits')
axes[1].tick_params(axis='x',rotation=60); axes[1].legend()
finish(fig,'02_dataset_distribution')
# %% [markdown]
# 3. Learning curves, perplexity, generalization gaps, and epoch time
# Positive validation-minus-training loss suggests worse performance on unseen text.
# Perplexity is exp(cross-entropy); lower is better. Timing reflects this local GPU run.
# %%
epochs=np.array([r['epoch'] for r in hist])
fig,axes=plt.subplots(2,3,figsize=(16,8))
for prefix,name in [('train','Training'),('val','Validation')]:
    loss=np.array([r[prefix+'_loss'] for r in hist]); acc=np.array([r[prefix+'_accuracy'] for r in hist])
    axes[0,0].plot(epochs,loss,label=name)
    axes[0,1].plot(epochs,100*acc,label=name)
    axes[0,2].plot(epochs,np.exp(loss),label=name)
for ax,title,y in zip(axes[0],['Cross-entropy','Character accuracy','Perplexity'],['Loss','Accuracy (%)','exp(loss)']):
    ax.set(title=title,ylabel=y); ax.legend(); ax.axvline(saved['epoch'],ls=':',color='gray',label='Best epoch')
axes[1,0].plot(epochs,[r['val_loss']-r['train_loss'] for r in hist],color='#b75d32')
axes[1,0].axhline(0,color='gray',lw=1); axes[1,0].set(title='Loss generalization gap',ylabel='Validation − training loss')
axes[1,1].plot(epochs,[100*(r['train_accuracy']-r['val_accuracy']) for r in hist],color='#8655a3')
axes[1,1].set(title='Accuracy generalization gap',ylabel='Training − validation (percentage points)')
axes[1,2].bar(epochs,[r['seconds'] for r in hist]); axes[1,2].set(title='Time per epoch',ylabel='Seconds')
for ax in axes.flat: ax.set_xlabel('Epoch')
finish(fig,'03_training_dashboard')
# %% [markdown]
# 4. Collect held-out predictions
# Teacher forcing supplies the real previous characters. Each window resets hidden
# state, matching the original evaluation. Top-k accuracy asks whether the true next
# character is among the k largest predicted probabilities.
# %%
tokens=torch.tensor([lookup.get(c,0) for c in parts['Test']],dtype=torch.long)
length=cfg['sequence_length']; n=(len(tokens)-1)//length
xs=tokens[:n*length].reshape(n,length)
ys=tokens[1:n*length+1].reshape(n,length)
probs=[]
with torch.inference_mode():
    for start in range(0,n,cfg['batch_size']):
        logits,_=net(xs[start:start+cfg['batch_size']].to(device))
        probs.append(logits.softmax(-1).cpu().numpy())
p=np.concatenate(probs)
truth=ys.numpy().ravel(); prob=p.reshape(-1,len(vocab)); pred=prob.argmax(1)
correct=pred==truth; confidence=prob.max(1)
assert abs(correct.mean()-metrics['test_accuracy'])<1e-8
cm=np.zeros((len(vocab),len(vocab)),dtype=np.int64)
np.add.at(cm,(truth,pred),1)
support=cm.sum(1)
recall=np.divide(cm.diagonal(),support,out=np.zeros(len(vocab)),where=support>0)
precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(len(vocab)),where=cm.sum(0)>0)
print(f'Analyzed {len(truth):,} next-character predictions; test accuracy {correct.mean():.2%}.')
# %% [markdown]
# 5. Baseline comparison and top-k prediction accuracy
# The baseline always emits the most common training character. Top-k values are
# calculated from the same held-out targets and are not free-running text accuracy.
# %%
fig,axes=plt.subplots(1,2,figsize=(12,4))
values=[100*metrics['baseline_accuracy'],100*correct.mean()]
axes[0].bar(['Most-common-character baseline','Simple RNN'],values,color=['#a7b1bb','#3267a8'])
axes[0].set(title='Held-out accuracy vs baseline',ylabel='Accuracy (%)',ylim=(0,100))
for i,v in enumerate(values): axes[0].text(i,v+2,f'{v:.2f}%',ha='center')
rank=np.argsort(prob,axis=1)[:,::-1]
topk=[100*(rank[:,:k]==truth[:,None]).any(1).mean() for k in [1,3,5,10]]
axes[1].bar(['Top 1','Top 3','Top 5','Top 10'],topk,color='#419e75')
axes[1].set(title='True next character included in top-k',ylabel='Accuracy (%)',ylim=(0,105))
for i,v in enumerate(topk): axes[1].text(i,v+1,f'{v:.1f}%',ha='center')
finish(fig,'04_accuracy_comparisons')
# %% [markdown]
# 6. Character confusion matrix
# Rows are actual characters; columns are predictions. Every supported test character
# is included. Rows are normalized to percentages so rare characters remain visible.
# %%
ids=np.flatnonzero(support)
normalized=cm[ids][:,ids]/support[ids,None]*100
fig,ax=plt.subplots(figsize=(15,13))
im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=100)
ax.set(xticks=np.arange(len(ids)),yticks=np.arange(len(ids)),
       xticklabels=[label(vocab[i]) for i in ids],yticklabels=[label(vocab[i]) for i in ids],
       xlabel='Predicted character',ylabel='Actual character',title='Held-out character confusion matrix — row percentages')
ax.tick_params(axis='x',rotation=90,labelsize=8); ax.tick_params(axis='y',labelsize=8); ax.grid(False)
fig.colorbar(im,ax=ax,label='Percentage of actual character (%)',shrink=.75)
finish(fig,'05_confusion_matrix')
# %% [markdown]
# 7. Per-character precision/recall and frequent errors
# Precision asks how often a predicted character is correct; recall asks how often an
# actual character is recovered. The 20 most common test characters are shown.
# Frequent error pairs use raw counts, so common characters naturally dominate.
# %%
frequent=np.argsort(support)[-20:][::-1]
errors=cm.copy(); np.fill_diagonal(errors,0)
pairs=[np.unravel_index(i,errors.shape) for i in np.argsort(errors.ravel())[-15:][::-1] if errors.ravel()[i]>0]
fig,axes=plt.subplots(1,2,figsize=(16,6))
positions=np.arange(len(frequent))
axes[0].barh(positions-.18,100*precision[frequent],.36,label='Precision')
axes[0].barh(positions+.18,100*recall[frequent],.36,label='Recall')
axes[0].set(yticks=positions,yticklabels=[f'{label(vocab[i])} (n={support[i]:,})' for i in frequent],xlabel='Percent',title='Per-character performance',xlim=(0,100))
axes[0].invert_yaxis(); axes[0].legend()
axes[1].barh(np.arange(len(pairs)),[errors[i,j] for i,j in pairs],color='#c47a42')
axes[1].set(yticks=np.arange(len(pairs)),yticklabels=[f'{label(vocab[i])} → {label(vocab[j])}' for i,j in pairs],xlabel='Incorrect predictions',title='Most frequent confusions: actual → predicted')
axes[1].invert_yaxis()
finish(fig,'06_character_errors')
# %% [markdown]
# 8. Confidence calibration and correctness
# A calibrated prediction with confidence 0.7 should be correct about 70% of the time.
# The reliability plot groups maximum softmax confidence into ten fixed bins.
# ECE is the count-weighted absolute confidence/accuracy gap for these bins.
# %%
edges=np.linspace(0,1,11); binid=np.minimum(np.digitize(confidence,edges)-1,9)
bin_stats=[]
for i in range(10):
    mask=binid==i
    if mask.any(): bin_stats.append((confidence[mask].mean(),correct[mask].mean(),int(mask.sum())))
ece=sum(abs(c-a)*n for c,a,n in bin_stats)/len(truth)
fig,axes=plt.subplots(1,3,figsize=(15,4))
axes[0].plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
axes[0].plot([r[0] for r in bin_stats],[r[1] for r in bin_stats],'o-',label='Model')
axes[0].set(title=f'Reliability diagram (ECE {ece:.3f})',xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1)); axes[0].legend()
axes[1].hist([confidence[correct],confidence[~correct]],bins=edges,label=['Correct','Incorrect'],color=['#419e75','#c47a42'])
axes[1].set(title='Confidence by correctness',xlabel='Maximum probability',ylabel='Prediction count'); axes[1].legend()
axes[2].bar(np.arange(10),[int((binid==i).sum()) for i in range(10)])
axes[2].set(xticks=np.arange(10),xticklabels=[f'{i/10:.1f}' for i in range(10)],title='Calibration bin support',xlabel='Confidence bin lower bound',ylabel='Prediction count')
finish(fig,'07_confidence_calibration')
# %% [markdown]
# 9. Accuracy by context position and prediction heatmap
# Position 1 sees only one input character after a hidden-state reset; later positions
# have more context. This is an observational curve, not a controlled sequence-length
# experiment. The heatmap shows the first held-out window's probability distributions.
# %%
fig,axes=plt.subplots(1,2,figsize=(16,5))
axes[0].plot(np.arange(1,length+1),100*correct.reshape(n,length).mean(0),alpha=.7)
axes[0].axhline(100*correct.mean(),ls='--',color='#c47a42',label='Overall test accuracy')
axes[0].set(title='Accuracy vs position within a window',xlabel='Input position (available context)',ylabel='Accuracy (%)'); axes[0].legend()
shown=40; candidates=np.argsort(prob.mean(0))[-20:][::-1]
im=axes[1].imshow(p[0,:shown,candidates],aspect='auto',cmap='magma',vmin=0,vmax=1)
axes[1].set(yticks=np.arange(len(candidates)),yticklabels=[label(vocab[i]) for i in candidates],xticks=np.arange(shown),xticklabels=[label(vocab[i]) for i in ys[0,:shown]],xlabel='Actual next character, first test window',ylabel='Candidate character',title='Next-character probability heatmap')
axes[1].tick_params(axis='x',rotation=90,labelsize=7); axes[1].grid(False)
fig.colorbar(im,ax=axes[1],label='Predicted probability')
finish(fig,'08_context_and_probabilities')
# %% [markdown]
# 10. Temperature changes the next-character distribution
# All three distributions use the same seed and checkpoint. Higher temperature
# flattens probabilities. The 12 leading candidates are shown; other probability
# mass is omitted from the bars. Entropy is computed over the full vocabulary.
# %%
seed='ROMEO: '
with torch.inference_mode(): logits,_=net(torch.tensor([[lookup[c] for c in seed]],device=device))
scores=logits[0,-1].detach().cpu().numpy(); scores[0]=-np.inf
temperatures=[.5,.8,1.2]
distributions=[]
for temp in temperatures:
    z=scores/temp; q=np.exp(z-np.max(z)); q/=q.sum(); distributions.append(q)
leading=np.argsort(distributions[1])[-12:][::-1]
fig,axes=plt.subplots(1,2,figsize=(13,4))
for j,(temp,q) in enumerate(zip(temperatures,distributions)):
    axes[0].bar(np.arange(12)+(j-1)*.25,q[leading],.25,label=f'T={temp}')
axes[0].set(xticks=np.arange(12),xticklabels=[label(vocab[i]) for i in leading],title=f'Next-character distribution after {seed!r}',ylabel='Probability'); axes[0].legend()
entropy=[-np.sum(q[q>0]*np.log2(q[q>0])) for q in distributions]
axes[1].plot(temperatures,entropy,'o-'); axes[1].set(title='Uncertainty vs temperature',xlabel='Temperature',ylabel='Entropy (bits)')
finish(fig,'09_temperature_distribution')
# %% [markdown]
# 11. Generated-text diversity and character composition
# Each temperature uses five fixed random seeds, producing 1,000 new characters per
# seed. Error bars show one standard deviation, not confidence intervals. Distinct
# trigrams are unique 3-character strings divided by all 3-character strings.
# More diversity does not necessarily mean better text. The reference uses five
# disjoint 1,000-character excerpts from held-out real text.
# %%
@torch.inference_mode()
def analysis_generate(temp,random_seed,count=1000):
    rng=torch.Generator(device=device).manual_seed(random_seed)
    x=torch.tensor([[lookup[c] for c in seed]],device=device)
    z,h=net(x); result=[]
    for _ in range(count):
        s=z[0,-1].clone()/temp; s[0]=-float('inf')
        token=torch.multinomial(s.softmax(-1),1,generator=rng)
        result.append(vocab[token.item()]); z,h=net(token.reshape(1,1),h)
    return ''.join(result)
def text_stats(t):
    counts=np.array(list(Counter(t).values()),dtype=float); q=counts/counts.sum()
    grams=[t[i:i+3] for i in range(len(t)-2)]
    return [-np.sum(q*np.log2(q)),len(set(grams))/len(grams),np.mean([a==b for a,b in zip(t,t[1:])])]
generated={f'T={temp}':[analysis_generate(temp,s) for s in range(5)] for temp in temperatures}
generated['Real text']=[parts['Test'][i*1000:(i+1)*1000] for i in range(5)]
fig,axes=plt.subplots(1,3,figsize=(15,4))
for j,(title,ylabel) in enumerate([('Character entropy','Bits'),('Distinct character trigrams','Ratio'),('Adjacent repeated characters','Fraction')]):
    stats=[np.array([text_stats(t) for t in texts])[:,j] for texts in generated.values()]
    axes[j].bar(list(generated),[v.mean() for v in stats],yerr=[v.std(ddof=1) for v in stats],capsize=4,color=['#3267a8','#419e75','#c47a42','#999999'])
    axes[j].set(title=title,ylabel=ylabel); axes[j].tick_params(axis='x',rotation=20)
finish(fig,'10_generated_text_diversity')
fig,ax=plt.subplots(figsize=(12,4))
for j,(name,texts) in enumerate(generated.items()):
    joined=''.join(texts); counts=Counter(joined)
    ax.bar(np.arange(len(common))+(j-1.5)*.2,[100*counts[c]/len(joined) for c in common],.2,label=name)
ax.set(xticks=np.arange(len(common)),xticklabels=[label(c) for c in common],title='Generated vs real character composition',ylabel='Frequency (%)')
ax.tick_params(axis='x',rotation=60); ax.legend()
finish(fig,'11_generated_character_distribution')
(OUT/'generation_analysis.json').write_text(json.dumps({'seed':seed,'characters_per_sample':1000,'samples':generated,'calibration_ece':float(ece)},indent=2),encoding='utf-8')
print('Saved all extended plots to',FIG)
# %% [markdown]
# Interpretation and limitations
# - Use validation curves to discuss generalization; use test plots only for reporting.
# - The confusion matrix and per-character scores reveal which symbols are difficult.
# - Confidence and top-k plots describe next-character prediction under teacher forcing.
# - Generation is free-running and can compound errors even when test accuracy is reasonable.
# - Temperature trades predictability for diversity; read samples to assess coherence.
# - This is a single training run. Curves do not establish statistical significance.
# - ROC curves, regression residuals, and object-detection plots are not appropriate
#   default diagnostics for this multiclass character-language-model project.
