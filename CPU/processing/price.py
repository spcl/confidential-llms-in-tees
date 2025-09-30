import matplotlib.pyplot as plt

# Define the data
machine_types = [
    'c4-highcpu-2', 'c4-highcpu-4', 'c4-highcpu-8',
    'c4-highcpu-16', 'c4-highcpu-32', 'c4-highcpu-48',
    'c4-highcpu-96', 'c4-highcpu-192'
]
vcpus = [2, 4, 8, 16, 32, 48, 96, 192]
prices = [0.085052, 0.170104, 0.340208, 0.680416, 1.360832, 2.041248, 4.082496, 8.164992]

# Compute vCPU per dollar for each machine type
vcpu_per_dollar = [v / p for v, p in zip(vcpus, prices)]
print("vCPU per Dollar ratio for each machine type:")
for machine, ratio in zip(machine_types, vcpu_per_dollar):
    print(f"{machine}: {ratio:.2f} vCPU/US$")

# Plot the ratios
plt.figure(figsize=(10, 6))
bars = plt.bar(machine_types, vcpu_per_dollar, color='skyblue')
plt.xlabel('Machine Type')
plt.ylabel('vCPU per Dollar per Hour')
plt.title('vCPU per Dollar for c4-highcpu Machines')
plt.xticks(rotation=45)

# Annotate each bar with the computed ratio
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Estimate the cost of 60 vCPUs
# (Since all machines share the same ratio, we can use any one, e.g., c4-highcpu-2)
cost_per_vcpu = prices[0] / vcpus[0]  # 0.085052 / 2
cost_for_60 = 60 * cost_per_vcpu
print(f"Estimated cost for 60 vCPUs: ${cost_for_60:.2f} per hour")
