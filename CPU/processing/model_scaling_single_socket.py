import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy import stats

# Get the directory from the first argument
file = sys.argv[1]
plt.rcParams.update({'font.size': 12}) 

def plot_arrows(start, end, height, height_diff, text, ax):
    ax.annotate('', xy=(start, height), xytext=(end, height), arrowprops=dict(arrowstyle="<-", lw=2), zorder=2)
    ax.text((start+end)/2-0.15, height+0.04*height_diff, text, ha='center', va='center', fontsize=10, backgroundcolor="w", zorder=1, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='white', lw=0))

def filter_dataframe(batch_size, throughput, data_type, order, size, numa):
    # Initial load and case filtering
    df = pd.read_csv(f"{file}")
    df = df.loc[df['index'] != 0]
    df = df.loc[df['system'].isin(order)]
    df = df.loc[df['bs'] == batch_size]
    df = df.loc[df['dt'] == data_type]
    df = df.loc[df['size'] == size]
    df = df.loc[df['numa'] == numa]
    if throughput:
        df["throughput"] = 6/df["time"]
    df['time'] *= 1000

    # Filter outliers
    s = 0
    for system in order:
        mask = np.abs(stats.zscore(df.loc[(df['system'] == system), 'time'])) < 3
        s += sum(~mask)/(sum(mask) + sum(~mask)) / len(order)
        df.drop(df[(df['system'] == system) & (~mask)].index, inplace=True)
    print(f"Filtered: {s*100}%")

    # Compute overhead dictionary
    overheads = {system: {system: 0 for system in order} for system in order}
    for system1 in order:
        for system2 in order:
            if throughput:
                overheads[system1][system2] = abs(1-df[df["system"] == system1]["throughput"].mean()/df[df['system'] == system2]['throughput'].mean())
            else:
                overheads[system1][system2] = abs(1-df[df["system"] == system1]["time"].mean()/df[df['system'] == system2]['time'].mean())
    print(overheads["TDX"]["VM"])
    return df, overheads

# Define the constants
ORDER = ["baremetal", "VM", "TDX", "SGX"]
COLORS = ['#1F8B87', '#76C1C0', '#76C1C0', '#d4e1e2']
COLORS2 = ['#76C1C0', '#76C1C0', '#1F8B87', '#d4e1e2']
NUMA = "single"

# Define variables
columns = [{"size": "7B", "datatype": "bf16"}, {"size": "13B", "datatype": "bf16"}, {"size": "7B", "datatype": "int8"}, {"size": "13B", "datatype": "int8"}]
rows = [{"batch_size": "6bs", "throughput": True}, {"batch_size": "1bs", "throughput": False}]
arrow_locations = [[[0.25, 0.75, 0.1, 0.9], [0.25, 0.75, 0.1, 0.91], [0.15, 0.15, 0.02, 0.9], [0.2, 0.7, 0.05, 0.9]],
                   [[0.45, 0.1, 0.7, 0.9], [0.5, 0.5, 0.7, 0.9], [0.45, 0.1, 0.7, 0.9], [0.5, 0.1, 0.7, 0.9]]]

# Define plot
fig, ax = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=(15, 5)) # 15x4 in the past
plt.subplots_adjust(wspace=0.2, hspace=0.1)
fig.suptitle("Single socket", y=0.96, x=0.51)

# Plot
for column_index, column in enumerate(columns):
    for row_index, row in enumerate(rows): 
        df, overheads = filter_dataframe(row["batch_size"], row["throughput"], column["datatype"], ORDER, column["size"], NUMA)
        colors = COLORS if column_index % 2 == 0 else COLORS2
        sns.violinplot(data=df, x="system", hue="system", y="throughput" if row["throughput"] else "time", inner="quart", order=ORDER, palette=colors, ax=ax[row_index][column_index], zorder=2, legend=False)
        for i in range(len(ORDER)):
            ax[row_index][column_index].axvline(x=i, color='gray', linestyle='--', zorder=0, alpha=0.5)
        ax[row_index][column_index].set_xlabel("")
        ax[row_index][column_index].set_ylabel("")
        ax[row_index][column_index].grid(axis="y")
        ax[row_index][column_index].set_axisbelow(True)
        if row_index != len(rows) - 1:
            ax[row_index][column_index].set_xticks([])
            ax[row_index][column_index].set_title(f"{column['size']} {column['datatype']}")
        ax[row_index][column_index].set_xticklabels(ax[row_index][column_index].get_xticklabels(), rotation=10, ha='right')
        y_diff = abs(ax[row_index][column_index].get_ylim()[1] - ax[row_index][column_index].get_ylim()[0])
        y_min = ax[row_index][column_index].get_ylim()[0]
        symbol = "\u2212" if row["throughput"] else "\u002B"
        plot_arrows(0, 1, y_min + arrow_locations[row_index][column_index][0] * y_diff, y_diff, f'{symbol}{overheads["VM"]["baremetal"]:.2%}', ax[row_index][column_index])
        plot_arrows(1, 2, y_min + arrow_locations[row_index][column_index][1] * y_diff, y_diff, f'{symbol}{overheads["TDX"]["VM"]:.2%}', ax[row_index][column_index])
        plot_arrows(0, 2, y_min + arrow_locations[row_index][column_index][2] * y_diff, y_diff, f'{symbol}{overheads["TDX"]["baremetal"]:.2%}', ax[row_index][column_index])
        plot_arrows(0, 3, y_min + arrow_locations[row_index][column_index][3] * y_diff, y_diff, f'{symbol}{overheads["SGX"]["baremetal"]:.2%}', ax[row_index][column_index])

ax[0][0].set_ylabel("Througput [tokens/s]")
ax[1][0].set_ylabel("Next token latency [ms]")
# fig.supxlabel('System type', fontsize=12)

plt.savefig("../figures/overall.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
plt.savefig("../figures/overall.svg", bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()