#!/usr/bin/env python3
import os
import re
import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────────────────────
CC_DIR       = "results_2025-05-04_16-34-55"
VM_DIR       = "results_2025-05-04_21-51-14"
FIXED_INPUT  = 128      # input length for the left plot
FIXED_BATCH  = 4        # batch size for the right plot
MAX_BATCH    = 512      # only include batch_size ≤ 512

# Plot styling
ORDER_B      = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
HUE_ORDER    = ["GPU", "cGPU"]            # first GPU then cGPU
COLORS       = ['#E07A5F', '#F4A261']  # match GPU, cGPU respectively
PATTERN      = re.compile(r"latency_in(\d+)_bs(\d+)\.json$")
plt.rcParams.update({'font.size': 12})
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load JSONs into a DataFrame
rows = []
for system, d in [("cGPU", CC_DIR), ("GPU", VM_DIR)]:
    for path in glob.glob(os.path.join(d, "latency_in*_bs*.json")):
        m = PATTERN.search(os.path.basename(path))
        if not m:
            continue
        inp = int(m.group(1))
        bs  = int(m.group(2))
        if bs > MAX_BATCH:
            continue

        with open(path) as f:
            data = json.load(f)

        latencies = data.get("latencies", [])
        # skip files with no per-sample latencies
        if not latencies:
            continue

        # append one row per-sample
        for lat in latencies:
            rows.append({
                "system":     system,
                "input_len":  inp,
                "batch_size": bs,
                "latency_ms": lat,
                "throughput": 128 * bs / lat  # samples per second
            })

df = pd.DataFrame(rows)

# 2) Prepare figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

# ── Left: throughput vs batch size (fixed input) ────────────────────────────────
df_b = df[df["input_len"] == FIXED_INPUT].copy()
df_b["batch_size"] = df_b["batch_size"].astype(str)
print(df_b)

sns.barplot(
    data=df_b,
    x="batch_size",
    y="throughput",
    hue="system",
    hue_order=HUE_ORDER,
    palette=COLORS,
    order=ORDER_B,
    ax=ax1,
    zorder=2
)

# vertical lines at each batch tick
for idx in range(len(ORDER_B)):
    ax1.axvline(x=idx, color='gray', linestyle='--', zorder=0, alpha=0.5)

# annotate cGPU bars with overhead relative to GPU
# patches are laid out [GPU@32, cGPU@32, GPU@64, cGPU@64, ...]
for idx, bar in enumerate(ax1.patches):
    # cGPU bars are at odd indices when HUE_ORDER=["GPU","cGPU"]
    if idx // len(ORDER_B) == 1:
        cc_h = bar.get_height()
        vm_h = ax1.patches[idx % len(ORDER_B)].get_height()
        if vm_h > 0:
            ov = cc_h / vm_h - 1
            x = bar.get_x() + bar.get_width() / 2
            if idx % len(ORDER_B) > 4:
                y = cc_h * 0.5 #cc_h * 1.03
                va = 'center'
            else:
                y = cc_h * 1.03
                va = 'bottom'
            ax1.annotate(f"{ov:.2%}", xy=(x, y),
                         ha='center', va=va, rotation='vertical')

ax1.set_title(f"input length={FIXED_INPUT}")
ax1.set_xlabel("Batch size")
ax1.set_ylabel("Throughput (tokens/sec)")
ax1.grid(axis='y')
ax1.set_axisbelow(True)
ax1.legend(title="")

# ── Right: throughput vs input length (fixed batch) ────────────────────────────
df_i = df[df["batch_size"] == FIXED_BATCH].copy()
# determine order of input lengths as strings
ORDER_I = sorted(df_i["input_len"].unique(), key=int)
ORDER_I = [str(i) for i in ORDER_I]
df_i["input_len"] = df_i["input_len"].astype(str)

sns.barplot(
    data=df_i,
    x="input_len",
    y="throughput",
    hue="system",
    hue_order=HUE_ORDER,
    palette=COLORS,
    order=ORDER_I,
    ax=ax2,
    zorder=2
)

# vertical lines at each input-size tick
for idx in range(len(ORDER_I)):
    ax2.axvline(x=idx, color='gray', linestyle='--', zorder=0, alpha=0.5)

# annotate cGPU bars with overhead relative to GPU
# patches are laid out [GPU@32, cGPU@32, GPU@64, cGPU@64, ...]
for idx, bar in enumerate(ax2.patches):
    # cGPU bars are at odd indices when HUE_ORDER=["GPU","cGPU"]
    if idx // len(ORDER_I) == 1:
        cc_h = bar.get_height()
        vm_h = ax2.patches[idx % len(ORDER_I)].get_height()
        if vm_h > 0:
            ov = cc_h / vm_h - 1
            x = bar.get_x() + bar.get_width() / 2
            if idx % len(ORDER_I) >= 0:
                y = cc_h * 0.5 #cc_h * 1.03
                va = 'center'
            else:
                y = cc_h * 1.03
                va = 'bottom'
            ax2.annotate(f"{ov:.2%}", xy=(x, y),
                         ha='center', va=va, rotation='vertical')

ax2.set_title(f"batch size={FIXED_BATCH}")
ax2.set_xlabel("Input length (tokens)")
ax2.set_ylabel("")
ax2.grid(axis='y')
ax2.set_axisbelow(True)
ax2.legend().remove()

# ── Final touches ───────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("throughput_comparison_vm_cc.pdf", bbox_inches='tight', transparent=True)
plt.show()
