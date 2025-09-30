import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy import stats

# Get the directory from the first argument
file = sys.argv[1]
plt.rcParams.update({'font.size': 12}) 

# Define the constants
ORDER = ["baremetal", "VM", "TDX"]
EXPERIMENTS = ["BM25 reranked", "BM25", "sbert"]
COLORS = ['#1F8B87', '#76C1C0', '#d4e1e2']

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6, 2.5))
df = pd.read_csv(f"{file}")
df = df[df["iteration"] >= 3].groupby(["experiment", "iteration", "system"]).mean().reset_index()

for i in range(3):
    sns.barplot(data=df[df["experiment"] == EXPERIMENTS[i]], x="system", y="time", order=ORDER, palette=COLORS, ax=ax[i])
    for index, p in enumerate(ax[i].patches):
        height = p.get_height()
        if index:
            location = (p.get_x() + p.get_width() / 2, height / 2)
            va = 'center'
            ax[i].annotate(f"{height/ax[i].patches[0].get_height()-1:.2%}",
                        xy=location,  # X and Y coordinates
                        ha='center',  # Horizontal alignment
                        va=va,  # Vertical alignment
                        rotation='vertical')
    if i == 0:
        ax[i].set_ylabel("Time (ms)")
    else:
        ax[i].set_ylabel("")
    ax[i].set_xlabel("")
    ax[i].grid(axis='y')
    ax[i].set_axisbelow(True)
    ax[i].set_title(EXPERIMENTS[i])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=10, ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.savefig("../figures/rag.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()