import glob
import os
import csv
import sys
import re

pattern = r"Iteration:\s*((?:[0-9]|\d{2,})),\s*Time:\s*([\d.]+)\s*sec"

# Write header and truncate existing file (if it exists)
with open(f"./results.csv", 'w', newline='') as results:
    result_first_writer = csv.writer(results)
    result_first_writer.writerow(["system", "numa", "vCPU", "bs", "dt", "in_size", "out_size", "model", "index", "time"])

# Process each directory provided in the arguments
for directory in sys.argv[1:]:
    # Process each .txt file found
    for txt_file in glob.glob(f"./{directory}/**/*.txt"):
        with open(txt_file, 'r') as file:
            print(txt_file)
            lines = file.readlines()

            # Process each line in the file
            for index, line in enumerate(lines, start=1):
                line = line.strip()

                match = re.search(pattern, line)

                if match:
                    # Convert line into an array of numbers
                    time = float(match.group(2))
                    iteration = int(match.group(1))
                    # Get file name without .txt extension
                    file_name = os.path.splitext(os.path.basename(txt_file))[0]
                    # Get parameters
                    system, in_size, out_size, vCPUs, numa, batch_size, model, data_type = file_name.split("-")
                    in_size = in_size.replace("in", "")
                    out_size = out_size.replace("out", "")
                    vCPUs = vCPUs.replace("vCPU", "")
                    model = model.upper()

                    # Write the results to results.csv
                    with open(f"./results.csv", 'a', newline='') as results:
                        result_first_writer = csv.writer(results)
                        result_first_writer.writerow([system, numa, vCPUs, batch_size, data_type, in_size, out_size, model, iteration, time])

                # Check if the line contains the specified condition
                if "token times:" in line:
                    tokens = True
                    index_zero = index