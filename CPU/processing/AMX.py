import seaborn as sns
import matplotlib.pyplot as plt
from AMX_latency import AMX_latency
from AMX_throughput import AMX_throughput

fig, ax = plt.subplots(2, 2, figsize=(12, 4.5))
AMX_throughput(fig, ax[0])
AMX_latency(fig, ax[1])

plt.tight_layout()
plt.subplots_adjust(wspace=0.11, hspace=0.05)
plt.savefig("../figures/AMX_combined.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()