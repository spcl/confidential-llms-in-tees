import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy import stats

# Get the directory from the first argument
file = sys.argv[1]
plt.rcParams.update({'font.size': 12}) 

def filter_dataframe(data_type, order, size, numa):
    # Initial load and case filtering
    df = pd.read_csv(f"{file}")

    df = df.loc[df['index'] != 0]
    df = df.loc[df['system'].isin(order)]
    df = df.loc[df['dt'] == data_type]
    print(df)
    df = df.loc[df['size'] == size]
    df = df.loc[df['numa'] == numa]

    return df

# Define the constants
HUE_ORDER = ["VM (AMX)", "TDX (AMX)", "VM (no AMX)", "TDX (no AMX)"]
ORDER = ["1", "2", "4", "8", "16", "32", "64", "128"] #, "256", "512"
COLORS = ['#2E3B3F', '#1F8B87', '#76C1C0', '#d4e1e2']
NUMA = "1s"

def AMX_throughput(fig, ax):
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    df_bf16 = filter_dataframe("bf16", HUE_ORDER, "7B", NUMA)
    df_bf16['bs'] = df_bf16['bs'].str[:-2]
    df_bf16 = df_bf16[df_bf16['in_size'] == 128]
    print(df_bf16)
    df_bf16['throughput'] = df_bf16['bs'].astype(int) / df_bf16['time']

    sns.barplot(data=df_bf16, x="bs", hue="system", y="throughput", hue_order=HUE_ORDER, order=ORDER, palette=COLORS, ax=ax[0])
    for index, p in enumerate(ax[0].patches):
        height = p.get_height()
        if height > 0 and index > len(ORDER) - 1:
            if index % len(ORDER) < 6 or index >= 2 * len(ORDER):
                location = (p.get_x() + p.get_width() / 2, height + 30)
                va = 'bottom'
            else:
                location = (p.get_x() + p.get_width() / 2, height / 2)
                va = 'center'

            # if index >= 3 * len(ORDER):
            #     reference = ax[0].patches[len(ORDER) * 2 + index % len(ORDER)].get_height()
            # else:
            reference = ax[0].patches[index % len(ORDER)].get_height()

            ax[0].annotate(f"{height/reference-1:.2%}",
                        xy=location,  # X and Y coordinates
                        ha='center',  # Horizontal alignment
                        va=va,  # Vertical alignment
                        size=10,
                        rotation='vertical') #                     fontsize=9,
                        #fontweight='bold'
    ax[0].set_ylabel("Throughput\n(tokens/s)")
    ax[0].grid(axis='y')
    ax[0].set_xlabel("")
    ax[0].set_title("bf16")
    ax[0].set_xticks([])
    ax[0].legend().remove()
    ax[0].set_axisbelow(True)
    for i in ORDER:
        ax[0].axvline(x=i, color='gray', linestyle='--', zorder=0, alpha=0.5)

    df_int8 = filter_dataframe("int8", HUE_ORDER, "7B", NUMA)
    df_int8['bs'] = df_int8['bs'].str[:-2]
    df_int8['throughput'] = df_int8['bs'].astype(int) / df_int8['time']

    sns.barplot(data=df_int8, x="bs", hue="system", y="throughput", hue_order=HUE_ORDER, order=ORDER, palette=COLORS, ax=ax[1])
    for index, p in enumerate(ax[1].patches):
        height = p.get_height()
        if height > 0 and index > len(ORDER) - 1:
            if index % len(ORDER) < 5 or index >= 2 * len(ORDER):
                location = (p.get_x() + p.get_width() / 2, height + 30)
                va = 'bottom'
            else:
                location = (p.get_x() + p.get_width() / 2, height / 2)
                va = 'center'

            # if index >= 3 * len(ORDER):
            #     reference = ax[1].patches[len(ORDER) * 2 + index % len(ORDER)].get_height()
            # else:
            reference = ax[1].patches[index % len(ORDER)].get_height()
            ax[1].annotate(f"{height/reference-1:.2%}",
                        xy=location,  # X and Y coordinates
                        ha='center',  # Horizontal alignment
                        va=va,  # Vertical alignment
                        size=10,
                        rotation='vertical')
    ax[1].set_ylabel("")
    ax[1].grid(axis='y')
    ax[1].set_xlabel("")
    ax[1].legend().remove()
    ax[1].set_title("int8")
    ax[1].set_xticks([])
    ax[1].set_axisbelow(True)
    for i in ORDER:
        ax[1].axvline(x=i, color='gray', linestyle='--', zorder=0, alpha=0.5)

    # fig.supxlabel("Batch size")
    # plt.tight_layout()
    # plt.savefig("../figures/amx_tput.pdf", bbox_inches='tight', transparent=True)
    # plt.show()