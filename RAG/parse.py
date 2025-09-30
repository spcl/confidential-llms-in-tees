#!/usr/bin/env python3
import sys
import re
import glob
import csv

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: {} <directory_name>".format(sys.argv[0]))
    directory_name = sys.argv[1]

    # Create CSV writer to stdout
    writer = csv.writer(sys.stdout)
    writer.writerow(["system", "experiment", "iteration", "time"])

    # Regex pattern: look for [<iteration>:<time>ms]
    # For example, matches "[0:5428.69ms]" and captures "0" and "5428.69"
    pattern = re.compile(r"(\d+):(\d+(?:\.\d+)?)ms")

    # Iterate over all .log files in the current directory.
    for log_file in glob.glob(f"{directory_name}/**/*.log"):
        system_name = log_file.split("/")[-2]
        experiment = log_file.split("/")[-1][:-4]
        with open(log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    iteration = match.group(1)
                    time_ms = match.group(2)
                    writer.writerow([system_name, experiment, iteration, time_ms])

if __name__ == "__main__":
    main()
