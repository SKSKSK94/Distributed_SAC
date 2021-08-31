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

    ax.set_xticklabels([''] + destination, fontdict=fontdict, rotation=0)
    ax.set_yticklabels([''] + source, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('attention', rotation=-90, va="bottom")

    plt.show()

#%%
# alphas = torch.tensor([[0.1590, 0.2047, 0.1970, 0.1301, 0.1431, 0.1661],
#         [0.1593, 0.2207, 0.2912, 0.1469, 0.0258, 0.1561],
#         [0.1484, 0.2247, 0.3406, 0.1364, 0.0101, 0.1398],
#         [0.1593, 0.2037, 0.1942, 0.1285, 0.1499, 0.1644],
#         [0.1602, 0.2032, 0.1926, 0.1270, 0.1547, 0.1624],
#         [0.1582, 0.2279, 0.2058, 0.1091, 0.1501, 0.1489],
#         [0.1594, 0.2036, 0.1939, 0.1282, 0.1509, 0.1640],
#         [0.1518, 0.2241, 0.3264, 0.1398, 0.0132, 0.1447],
#         [0.1596, 0.2036, 0.1936, 0.1279, 0.1517, 0.1637],
#         [0.1590, 0.2047, 0.1970, 0.1301, 0.1431, 0.1661]])

# alphas = torch.tensor([[1.5267e-01, 9.7491e-02, 8.9942e-04, 5.1491e-02, 4.8151e-01, 2.1594e-01],
#         [1.5322e-01, 6.8293e-02, 1.0548e-01, 1.4257e-01, 3.3885e-01, 1.9158e-01],
#         [1.4206e-01, 1.5654e-01, 1.4199e-01, 2.1584e-01, 1.6203e-01, 1.8155e-01],
#         [7.9297e-02, 7.2261e-02, 1.4439e-03, 1.4212e-01, 3.9309e-01, 3.1179e-01],
#         [1.0057e-01, 9.3442e-02, 3.6232e-03, 1.2244e-01, 3.8633e-01, 2.9360e-01],
#         [8.6151e-02, 2.7272e-02, 3.4428e-04, 7.2103e-02, 5.0093e-01, 3.1320e-01],
#         [9.5981e-02, 1.0241e-01, 2.5874e-03, 1.2503e-01, 3.7343e-01, 3.0056e-01],
#         [9.4119e-02, 2.3704e-01, 2.2146e-01, 2.9103e-01, 4.7545e-02, 1.0880e-01],
#         [1.2428e-01, 5.5946e-02, 6.0919e-03, 1.8679e-01, 3.5949e-01, 2.6740e-01],
#         [1.0507e-01, 5.2097e-02, 2.1451e-03, 1.4332e-01, 3.7944e-01, 3.1794e-01]])

alphas = torch.tensor([[0.1338, 0.0721, 0.1265, 0.3266, 0.1823, 0.1588],
        [0.0732, 0.1065, 0.1686, 0.2095, 0.3251, 0.1171],
        [0.0751, 0.1074, 0.1548, 0.2082, 0.3326, 0.1219],
        [0.0910, 0.1214, 0.1075, 0.3017, 0.2537, 0.1247],
        [0.0976, 0.0702, 0.0616, 0.4770, 0.2023, 0.0913],
        [0.0696, 0.0925, 0.0284, 0.5423, 0.2126, 0.0545],
        [0.1071, 0.0714, 0.1363, 0.3290, 0.2402, 0.1159],
        [0.0547, 0.1138, 0.1750, 0.2251, 0.3294, 0.1020],
        [0.0651, 0.0764, 0.1637, 0.2740, 0.3326, 0.0883],
        [0.0558, 0.0725, 0.1760, 0.2740, 0.2931, 0.1285]])

alphas = alphas.detach().numpy()

target_labels = ['Enc 0', 'Enc 1', 'Enc 2', 'Enc 3', 'Enc 4', 'Enc 5']
source_labels = [ 
    "0. reach : Reach a goal position.",
    "1. push : Push the puck to a goal.",
    "2. pick-place : Pick and place a puck to a goal.",
    "3. door-open : Open a door with a revolving joint.",
    "4. drawer-open : Open a drawer.",
    "5. drawer-close : Push and close a drawer.",
    "6. button-press-topdown : Press a button from the top.",
    "7. peg-insert-side : Insert a peg sideways.",
    "8. window-open : Push and open a window.",
    "9. window-close : Push and close a window"
]
plot_attention(alphas, source_labels, target_labels)

# %%
