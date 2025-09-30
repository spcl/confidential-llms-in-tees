# deepspeed --bind_cores_to_rank  distributed/run_generation_with_deepspeed.py --deployment-mode --benchmark -m meta-llama/Llama-2-7b-hf  --ipex --batch-size 4 --profile --num-iter 15 --num-warmup 5 --max-new-tokens 128 --input-tokens 128 --token-latency --greedy

import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

def parse_dataframe(trace):
    trace_events = trace['traceEvents']
    forward_idx = 0
    modules = {
        "forward_idx": [],
        "name": [],
        "level_1": [],
        "level_2": [],
        "level_3": [],
        "level_4": [],
        "level_5": [],
        "level_6": [],
        "duration": []
    }
    current_level = [{'name': "", 'start': 0} for _ in range(6)]
    current_start = 0
    current_end = 0

    def save_to_modules(forward_idx, current_level, event):
        modules['forward_idx'].append(forward_idx - 1)
        if len(current_level) > 0:
            modules['duration'].append(event['ts'] - current_level[len(current_level) - 1]['start'])
        else:
            modules['duration'].append(event['dur'])
        modules['name'].append(event['name'])
        modules['level_1'].append(current_level[0]['name'] if len(current_level) > 0 else "")
        modules['level_2'].append(current_level[1]['name'] if len(current_level) > 1 else "")
        modules['level_3'].append(current_level[2]['name'] if len(current_level) > 2 else "")
        modules['level_4'].append(current_level[3]['name'] if len(current_level) > 3 else "")
        modules['level_5'].append(current_level[4]['name'] if len(current_level) > 4 else "")
        modules['level_6'].append(current_level[5]['name'] if len(current_level) > 5 else "")

    for event in trace_events:
        if event['ph'] == 'X':
            if event['name'] == 'forward':
                print(f"[{forward_idx}] Forward event: {event['ts']} - {event['dur']}")
                forward_idx += 1
                current_level = [{'name': "", 'start': 0} for _ in range(6)]
                current_start = event['ts']
                current_end = event['ts'] + event['dur']
                save_to_modules(forward_idx, [], event)
            elif event['ts'] >= current_start and event['ts'] <= current_end:
                if 'Module Hierarchy' in event['args']:
                    event_modules = event['args']['Module Hierarchy'].split('.')
                    for i, new in reversed(list(enumerate(event_modules))):
                        if i < 6:
                            if current_level[i]['name'] == "":
                                current_level[i]['name'] = new
                                current_level[i]['start'] = event['ts']
                            elif current_level[i]['name'] != new:
                                save_to_modules(forward_idx, current_level[:i+1], event)
                                current_level[i]['name'] = new
                                current_level[i]['start'] = event['ts']

    df = pd.DataFrame(modules)
    print(df)
    df.to_csv('parsed_trace.csv', index=False)
    return df

with open(sys.argv[1], 'r') as f:
    trace = json.load(f)
df_base = parse_dataframe(trace)
df_base['system'] = 'baremetal'

with open(sys.argv[2], 'r') as f:
    trace = json.load(f)
df_tdx = parse_dataframe(trace)
df_tdx['system'] = 'tdx'

df = pd.concat([df_base, df_tdx], ignore_index=True)

for i in range(1, 5):
    df = df[df[f'level_{i}'] != ""]
for i in range(5, 7):
    df = df[df[f'level_{i}'] == ""]
df = df[df['forward_idx'] != 0]
print(df)

grouped = (
    df.groupby(['level_4', 'system'], observed=True, dropna=False)['duration']
      .mean()
      .rename('total_duration')
      .reset_index()
)
df['total_duration'] = df['duration']
grouped = df
print(grouped)

# forward_avg_duration = df[df['name'] == 'forward']['duration'].mean()
# print(f"Average duration for name=='forward': {forward_avg_duration}")

# sum_total_duration_nonempty = grouped[grouped['level_3'] != ""]['total_duration'].sum()
# print(f"Sum of total_duration for rows with level_3 != '': {sum_total_duration_nonempty}")

# sum_total_duration_ipexdecoder = grouped[grouped['level_3'].str.contains("IPEXDecoderLayer", na=False)]['total_duration'].sum()
# print(f"Sum of total_duration for rows with level_3 including 'IPEXDecoderLayer': {sum_total_duration_ipexdecoder}")

# avg_total_duration_nonempty = grouped[grouped['level_3'] != ""]['total_duration'].mean()
# print(f"Average total_duration for rows with level_3 != '': {avg_total_duration_nonempty}")

# avg_total_duration_ipexdecoder = grouped[grouped['level_3'].str.contains("IPEXDecoderLayer", na=False)]['total_duration'].mean()
# print(f"Average total_duration for rows with level_3 including 'IPEXDecoderLayer': {avg_total_duration_ipexdecoder}")

# grouped = (
#     df.groupby(['level_4'], observed=True, dropna=False)['duration']
#       .mean()
#       .rename('total_duration')
#       .reset_index()
# )
# grouped["percentage"] = grouped['total_duration'] / avg_total_duration_ipexdecoder * 100
# print(grouped)

# Calculate percentage for each group relative to the sum for each system
grouped['percentage'] = grouped.groupby('system')['total_duration'].transform(lambda x: x / x.sum() * 100)
ORDER = ["input_layernorm(_IPEXRMSNormCPU)::forward", "self_attn(_IPEXAttentionCPU)::forward", "mha_linear_add(_IPEXlinearAddCPU)::forward", "post_attention_layernorm(_IPEXRMSNormCPU)::forward", "linear_silu_mul(_IPEXlinearSiluMulCPU)::forward", "mlp_linear_add(_IPEXlinearAddCPU)::forward"]
COLORS = ['#1F8B87', '#d4e1e2']

# Plot with seaborn: barplot for total_duration, hue by 'system'
plt.figure(figsize=(5, 3.25))
grouped['system'][grouped['system'] == "tdx"] = grouped['system'][grouped['system'] == "tdx"].str.upper()
sns.barplot(
    data=grouped,
    x='level_4',
    y='total_duration',
    hue='system',
    order=ORDER,
    palette=COLORS
)
plt.xlabel('')
# Modify x-tick labels: remove '::forward' and replace '_' with ' '
ax = plt.gca()
labels = [label.get_text().replace('::forward', '').replace('(_IPEX', '\n(IPEX').replace('_', ' ') for label in ax.get_xticklabels()]
ax.set_xticklabels(labels)
ax.set_axisbelow(True)

for index, p in enumerate(ax.patches):
        height = p.get_height()
        if height > 0 and index > len(ORDER) - 1:
            if index % len(ORDER) != 1 and index % len(ORDER) != 4:  # not self_attn and not linear_silu_mul
                location = (p.get_x() + p.get_width() / 2, height + 40)
                va = 'bottom'
                color = 'white'
            else:
                location = (p.get_x() + p.get_width() / 2, height / 2)
                va = 'center'
                color = 'none'
            ax.annotate(f"{height/ax.patches[index % len(ORDER)].get_height()-1:.2%}",
                        xy=location,  # X and Y coordinates
                        ha='center',  # Horizontal alignment
                        va=va,  # Vertical alignment
                        rotation='vertical',
                        bbox=dict(boxstyle='square,pad=0', fc=color, ec='none'))

plt.ylabel('Total Duration Per\nDecoder Block [us]')
plt.legend(title="")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('layer_duration.pdf', bbox_inches='tight')
plt.show()
