#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plot_attention(attention, source, destination):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(attention, cmap='viridis')

    for i in range(len(source)):
        for j in range(len(destination)):
          text = ax.text(j, i, "{:.2f}".format(attention[i, j]),
                       ha="center", va="center", color="w")

    fontdict = {'fontsize': 14}

    # ax.set_yticklabels(target_labels, minor=False)

    ax.set_xticklabels([''] + destination, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + source, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom")

    plt.show()

#%%
from distributed_CARE_src.utils import cfg_read
mt10_pretrained_embedding = cfg_read('../metadata/mt10_pretrained_embedding_my.json')
mt10_ordered_task_name = cfg_read('../metadata/mt10_ordered_task_name.json')

mt10_pretrained_embedding_list = []

for task_name in mt10_ordered_task_name:
    mt10_pretrained_embedding_list.append(
        mt10_pretrained_embedding[task_name]
    )
mt10_pretrained_embedding_ = torch.tensor(mt10_pretrained_embedding_list)
print(mt10_pretrained_embedding_.shape)
mt10_pretrained_embedding_ = mt10_pretrained_embedding_ / mt10_pretrained_embedding_.norm(dim=-1, keepdim=True)

cosine_similarity = torch.matmul(mt10_pretrained_embedding_, mt10_pretrained_embedding_.t())

source_labels = [ 
    "0. reach",
    "1. push",
    "2. pick-place",
    "3. door-open",
    "4. drawer-open",
    "5. drawer-close",
    "6. button-press-topdown",
    "7. peg-insert-side",
    "8. window-open",
    "9. window-close"
]
target_labels = source_labels

plot_attention(cosine_similarity, target_labels, source_labels)
