import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy import stats
from matplotlib.gridspec import GridSpec

# Get the directory from the first argument
file = sys.argv[1]
plt.rcParams.update({'font.size': 12}) 

def filter_dataframe(data_type, order, size, input_size):
    # Initial load and case filtering
    df = pd.read_csv(f"{file}")

    # df = df.loc[df['index'] != 0]
    df = df.loc[df['system'].isin(order)]
    df = df.loc[df['dt'] == data_type]
    print(df)
    df = df.loc[df['size'] == size]
    df = df.loc[df['in_size'] == input_size]

    return df

# Define the constants
HUE_ORDER = ["baremetal", "VM", "TDX"]
ORDER = ["2", "4", "8", "16", "32", "48", "60"] #, "256", "512"
INPUTS = [256, 512, 1024, 2048]
COLORS = ['#1F8B87', '#76C1C0', '#d4e1e2']

rows = 2
columns = 4
fig = plt.figure(figsize=(15, 4))
gs = GridSpec(rows, columns, height_ratios=[1, 1])
axes = []
for row in range(rows):
    column = []
    for col in range(columns):
        ax = fig.add_subplot(gs[row, col])
        column.append(ax)
        if row == 2:
            ax.axis("off")
    axes.append(column)

for index, input_size in enumerate(INPUTS):
    row = 0 if index < 4 else 3
    column = index % 4
    ax = [axes[row][column], axes[row + 1][column]]

    df_bf16 = filter_dataframe("bf16", HUE_ORDER, "7B", input_size)
    df_bf16['bs'] = df_bf16['bs'].str[:-2]
    df_bf16['throughput'] = df_bf16['bs'].astype(int) / df_bf16['time'] * 128

    sns.barplot(data=df_bf16, x="vCPU", hue="system", y="throughput", hue_order=HUE_ORDER, order=ORDER, palette=COLORS, ax=ax[0])
    for index, p in enumerate(ax[0].patches):
        height = p.get_height()
        if height > 0 and index > 2*len(ORDER) - 1:
            if index % len(ORDER) < 3:
                location = (p.get_x() + p.get_width() / 2, height * 1.2)
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
                        rotation='vertical',
                        fontsize=10) #                     fontsize=9,
                        #fontweight='bold'

    for i in ORDER:
        ax[0].axvline(x=i, color='gray', linestyle='--', zorder=0, alpha=0.5)
    ax[0].grid(axis='y')
    ax[0].set_xlabel("")
    ax[0].set_title(f"input size = {input_size}")
    ax[0].legend().remove()
    ax[0].set_xticks([])
    # ax[0].set_title("bf16")

    if column == 0:
        ax[0].set_ylabel("Throughput\n (tokens/s)")
    else: 
        ax[0].set_ylabel("")
    ax[0].set_axisbelow(True)
    ax[0].set_xticklabels([])

    df_emr = filter_dataframe("bf16", HUE_ORDER, "7B", input_size)
    df_emr['bs'] = df_emr['bs'].str[:-2]
    df_emr['throughput'] = df_emr['bs'].astype(int) / df_emr['time'] * 128
    memory = 128
    df_emr['cost_emr'] = 1e6 * (df_emr['vCPU'].astype(int) * 0.01152 + memory * 0.001309) / df_emr['throughput'] / 3600
    df_emr['cost_spr'] = 1e6 * (df_emr['vCPU'].astype(int) * 0.00604 + memory * 0.000808) / (df_emr['throughput'] * 0.75) / 3600
    gpu_cc_cost = {(128, 1): 13.966463202799583,
 (128, 2): 7.0308950161220105,
 (128, 4): 3.5766144538075766,
 (128, 8): 1.914538963106622,
 (128, 16): 1.085254798191451,
 (128, 32): 0.6428086263080585,
 (128, 64): 0.4383705951320425,
 (128, 128): 0.3480842650735822,
 (128, 256): 0.3291661763468436,
 (128, 512): 0.3272333761566251,
 (128, 1024): 0.32791145530846955,
 (256, 4): 1.833550928931185,
 (512, 4): 0.9742609679397212,
 (1024, 1): 1.817995655649457,
 (1024, 2): 0.9699529011092295,
 (1024, 4): 0.539056484187981,
 (1024, 8): 0.33382415631547707,
 (1024, 16): 0.22947454554873808,
 (1024, 32): 0.18255588023164052,
 (1024, 64): 0.1596774978751116,
 (1024, 128): 0.1554306064120403,
 (1024, 256): 0.1559189074043444,
 (1024, 512): 0.15637316513851668,
 (2048, 1): 0.9690387555178063,
 (2048, 2): 0.5396735087845624,
 (2048, 4): 0.32644508181175147,
 (2048, 8): 0.2240714243559176,
 (2048, 16): 0.17562872225547901,
 (2048, 32): 0.15259526991702324,
 (2048, 64): 0.14345381150493428,
 (2048, 128): 0.14700633658590015,
 (2048, 256): 0.14741199750539485,
 (2048, 512): 0.1478471349604274}
    gpu_raw_cost = {(128, 1): 12.925900934960325,
 (128, 2): 6.476382341412953,
 (128, 4): 3.332401754930449,
 (128, 8): 1.7781952921561066,
 (128, 16): 1.0089535299286823,
 (128, 32): 0.6125466076270812,
 (128, 64): 0.41673646708746226,
 (128, 128): 0.33110839402062603,
 (128, 256): 0.31074009263987157,
 (128, 512): 0.31295790816246816,
 (256, 4): 1.7147250117333697,
 (512, 4): 0.9106762901377997,
 (1024, 1): 1.6943122509108268,
 (1024, 2): 0.9011934738062516,
 (1024, 4): 0.5091571016330073,
 (1024, 8): 0.31566569140409245,
 (1024, 16): 0.2176320696649172,
 (1024, 32): 0.17113347980275245,
 (1024, 64): 0.14954864597865028,
 (1024, 128): 0.15037783438540914,
 (1024, 256): 0.15037909887610484,
 (1024, 512): 0.15032330964043286,
 (2048, 1): 0.9006105823638355,
 (2048, 2): 0.5048274023218947,
 (2048, 4): 0.3096166570932475,
 (2048, 8): 0.21290006777286094,
 (2048, 16): 0.1646925333803006,
 (2048, 32): 0.14421366860619753,
 (2048, 64): 0.13689581785634264,
 (2048, 128): 0.14049267049311676,
 (2048, 256): 0.14153718563235057,
 (2048, 512): 0.14125664647053526}

    sns.barplot(data=df_emr, x="vCPU", hue="system", y="cost_emr", hue_order=HUE_ORDER, order=ORDER, palette=COLORS, ax=ax[1])
    ax[1].axhline(y=gpu_cc_cost[(input_size, 4)], color='#F4A261', label="confidential H100 (cGPU)", linewidth=3)
    # ax[1].axhline(y=gpu_raw_cost[(input_size, 4)], color='red', label="non-confidential H100 (GPU)")
    relative_GPU = gpu_cc_cost[(input_size, 4)] / gpu_raw_cost[(input_size, 4)] - 1
    min_cost = df_emr[df_emr['system'] == "TDX"]['cost_emr'].min()
    relative_TDX = gpu_cc_cost[(input_size, 4)] / min_cost - 1
    ax[1].text(0.5, 0.9, f"ΔTDX={relative_TDX:.2%}", transform=ax[1].transAxes, va="top", ha="center", fontsize=10, backgroundcolor='white', color='black', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
# ΔGPU={relative_GPU:.2%}\n 
    # for index, p in enumerate(ax[1].patches):
    #     height = p.get_height()
    #     if height > 0 and index > 2*len(ORDER) - 1:
    #         if index % len(ORDER) < 5 or index >= 2 * len(ORDER):
    #             location = (p.get_x() + p.get_width() / 2, height + 30)
    #             va = 'bottom'
    #         else:
    #             location = (p.get_x() + p.get_width() / 2, height / 2)
    #             va = 'center'
            
    #         # if index >= 3 * len(ORDER):
    #         #     reference = ax[1].patches[len(ORDER) * 2 + index % len(ORDER)].get_height()
    #         # else:
    #         reference = ax[1].patches[index % len(ORDER)].get_height()
    #         # ax[1].annotate(f"{height/reference-1:.2%}",
    #         #             xy=location,  # X and Y coordinates
    #         #             ha='center',  # Horizontal alignment
    #         #             va=va,  # Vertical alignment
    #         #             rotation='vertical')
    ax[1].set_ylabel("")
    ax[1].grid(axis='y')
    ax[1].set_xlabel("")
    ax[1].legend().remove()
    for i in ORDER:
        ax[1].axvline(x=i, color='gray', linestyle='--', zorder=0, alpha=0.5)

    if column == 0:
        ax[1].set_ylabel("Estimated cost\n ($/million tokens)", labelpad=0)
    else: 
        ax[1].set_ylabel("")
    ax[1].set_axisbelow(True)

    # if row != len(BS) - 1:
    #     ax[1].set_xticklabels([])
    ax[1].set_axisbelow(True)
    if row != 0 or column != 0:
        ax[1].legend().remove()
    else:
        ax[1].legend(loc='lower left', ncol=5, bbox_to_anchor=(0, 2.20))
            

    # sns.barplot(data=df_emr, x="numa", hue="system", y="cost_spr", hue_order=HUE_ORDER, order=ORDER, palette=COLORS, ax=ax[2], ci="sd")
    # ax[2].axhline(y=gpu_cost[BS[row][:-2]], color='green')
    # for index, p in enumerate(ax[2].patches):
    #     height = p.get_height()
    #     if height > 0 and index > len(ORDER) - 1:
    #         if index % len(ORDER) < 5 or index >= 2 * len(ORDER):
    #             location = (p.get_x() + p.get_width() / 2, height + 30)
    #             va = 'bottom'
    #         else:
    #             location = (p.get_x() + p.get_width() / 2, height / 2)
    #             va = 'center'

    #         # if index >= 3 * len(ORDER):
    #         #     reference = ax[1].patches[len(ORDER) * 2 + index % len(ORDER)].get_height()
    #         # else:
    #         reference = ax[2].patches[index % len(ORDER)].get_height()
    #         # ax[2].annotate(f"{height/reference-1:.2%}",
    #         #             xy=location,  # X and Y coordinates
    #         #             ha='center',  # Horizontal alignment
    #         #             va=va,  # Vertical alignment
    #         #             rotation='vertical')
    # ax[2].set_ylabel("")
    # ax[2].grid(axis='y')
    # ax[2].set_xlabel("")
    # if row:
    #     ax[2].legend().remove()
    # else:
    #     ax[2].set_title("SPR spot (gen 4)")

    # if row == len(BS) / 2:
    #     ax[2].set_ylabel("Estimated cost ($/token)")
    # else: 
    #     ax[2].set_ylabel("")
    # ax[2].set_axisbelow(True)

    # if row != len(BS) - 1:
    #     ax[2].set_xticklabels([])
    # ax[2].legend().remove()
    # ax[2].set_axisbelow(True)

# if row == 0 or row == 4:
#     fig.supxlabel("vCPU size", y=0.05)

fig.text(0.51, 0.0, "Number of vCPUs", ha="center", va="center", fontsize=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05, wspace=0.15)
plt.savefig(f"../figures/vCPUs_GPU_EMR_inputs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()