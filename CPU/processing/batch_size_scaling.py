import seaborn as sns
import matplotlib.pyplot as plt
from batch_size_latency import batch_size_latency
from batch_size_throughput import batch_size_throughput 

fig, ax = plt.subplots(2, 2, figsize=(12, 4))
batch_size_throughput(fig, ax[0])
batch_size_latency(fig, ax[1])

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.savefig("../figures/batch_scaling_combined.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()