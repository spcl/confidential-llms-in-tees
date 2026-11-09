"""Turn a vLLM latency sweep into cost per million generated tokens.

Usage:  python parse.py <results_directory> (--sku <sku> | --cost <usd_per_hour>)

The printed dictionary is meant to be pasted into the gpu_cc_cost / gpu_raw_cost
constants of CPU/processing/vCPUs_batch_size.py and vCPUs_input.py.
"""

import argparse
import glob
import json
import os
import pprint
import re
import sys

# benchmark_latency.py measures the wall time of one generate() call that
# produces exactly --output-len tokens per sequence (ignore_eos=True), so the
# tokens produced per run are output_len * batch_size. The sweep does not record
# the output length in the file name, so it has to match benchmark_vllm.sh.
DEFAULT_OUTPUT_LEN = 128

# Azure list price in $/hour per SKU. The confidential VM and its
# non-confidential counterpart are priced differently, and a cost figure is only
# meaningful together with the machine it was measured on, so the price has to be
# given explicitly rather than defaulted.
VM_PRICES = {
    # Azure Retail Prices API, West Europe, Linux, priceType=Consumption, with the
    # Spot and Low Priority meters excluded (a Low Priority row is easy to mistake
    # for the on-demand rate if meterName is not checked).
    "NCC40ads_H100_v5": 8.90,  # confidential (cGPU), effective 2024-12-01
    "NC40ads_H100_v5": 9.08,   # non-confidential (GPU), effective 2024-03-01
}

# Pattern: latency_in{input_len}_bs{batch}.json
PATTERN = re.compile(r"latency_in(\d+)_bs(\d+)\.json$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="directory produced by benchmark_vllm.sh")
    price = parser.add_mutually_exclusive_group(required=True)
    price.add_argument("--sku", choices=sorted(VM_PRICES),
                       help="VM the results were measured on, priced from the table in this file")
    price.add_argument("--cost", type=float,
                       help="VM price in $/hour, for a SKU or region not in the table")
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN,
                        help=f"tokens generated per sequence (default {DEFAULT_OUTPUT_LEN})")
    args = parser.parse_args()

    cost = args.cost if args.cost is not None else VM_PRICES[args.sku]
    # to stderr, so stdout stays a dictionary that can be pasted as-is
    origin = f"{args.sku} " if args.sku else ""
    print(f"pricing {args.results_dir} at ${cost:.2f}/hour {origin}"
          f"({args.output_len} generated tokens per sequence)", file=sys.stderr)

    cost_by_input_bs = {}

    for path in sorted(glob.glob(os.path.join(args.results_dir, "latency_in*_bs*.json"))):
        match = PATTERN.search(os.path.basename(path))
        if not match:
            continue

        input_len = int(match.group(1))
        batch_size = int(match.group(2))

        with open(path, "r") as f:
            data = json.load(f)

        avg_latency = data.get("avg_latency")
        if avg_latency is None:
            print(f"skipping {path}: no avg_latency")
            continue

        # Throughput counts generated tokens, matching the CPU side
        # (bs / next-token-latency * output tokens) and plot_GPUs.py. Using the
        # input length here instead would overstate throughput by
        # input_len / output_len and understate the cost by the same factor.
        tokens_per_sec = args.output_len * batch_size / avg_latency
        cost_per_million = cost / (tokens_per_sec * 3600) * 1e6

        cost_by_input_bs[(input_len, batch_size)] = cost_per_million

    pprint.pprint(cost_by_input_bs)


if __name__ == "__main__":
    main()
