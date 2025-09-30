import json
import glob
import re
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python script.py <results_directory>")
    sys.exit(1)

results_dir = sys.argv[1]
COST = 6.98  # $/hour

latency_by_input_bs = {}

# Pattern: latency_in{input_len}_bs{batch}.json
pattern = re.compile(r"latency_in(\d+)_bs(\d+)\.json$")

for path in glob.glob(os.path.join(results_dir, "latency_in*_bs*.json")):
    match = pattern.search(os.path.basename(path))
    if not match:
        continue

    input_len = int(match.group(1))
    batch_size = int(match.group(2))

    with open(path, "r") as f:
        data = json.load(f)

    avg_latency = data.get("avg_latency")
    if avg_latency is None:
        continue

    # Cost per million tokens
    tokens_per_sec = input_len / avg_latency * batch_size
    cost_per_million = COST / (tokens_per_sec * 3600) * 1e6

    latency_by_input_bs[(input_len, batch_size)] = cost_per_million

# Pretty-print
import pprint
pprint.pprint(latency_by_input_bs)