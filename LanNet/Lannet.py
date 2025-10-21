import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. LanNet Architecture
# ----------------------------
class LanNet(nn.Module):
    def __init__(self):
        super(LanNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2*2*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ----------------------------
# 2. Dual-mode write time function
# ----------------------------
def dual_mode_write_time(num_bits, word_size_bits, slow_ns=20000, fast_ns=500, access_ns=5, weight_slow=0.75, weight_fast=0.25):
    """
    Calculates weighted average write time per word
    """
    num_words = num_bits / word_size_bits
    t_word_s = weight_slow * (access_ns*1e-9 + slow_ns*1e-9) + weight_fast * (access_ns*1e-9 + fast_ns*1e-9)
    t_total_ms = num_words * t_word_s * 1000  # convert s to ms
    return t_total_ms

# ----------------------------
# 3. LanNet Layer info (Output dimensions)
# ----------------------------
lan_layer_info = [
    {"name": "Input", "H": 32, "W": 32, "D": 3},
    {"name": "Conv1", "H": 30, "W": 30, "D": 16},
    {"name": "Pool1", "H": 15, "W": 15, "D": 16},
    {"name": "Conv2", "H": 13, "W": 13, "D": 32},
    {"name": "Pool2", "H": 6, "W": 6, "D": 32},
    {"name": "Conv3", "H": 4, "W": 4, "D": 64},
    {"name": "Pool3", "H": 2, "W": 2, "D": 64},
    {"name": "Flatten", "H": 1, "W": 1, "D": 256},
    {"name": "FC1", "H": 1, "W": 1, "D": 128},
    {"name": "FC2", "H": 1, "W": 1, "D": 10},
]

# ----------------------------
# 4. Compute per-layer write times
# ----------------------------
word_sizes = [16, 32, 64]  # in bits
lan_results = []

for layer in lan_layer_info:
    # Assume each element is 32-bit base for counting total bits
    num_bits = layer["H"] * layer["W"] * layer["D"] * 32
    row = {"Layer": layer["name"], "Output Dim": f"{layer['H']}x{layer['W']}x{layer['D']}"}
    for w in word_sizes:
        t_ms = dual_mode_write_time(num_bits=num_bits, word_size_bits=w)
        row[f"{w}-bit Write Time (ms)"] = round(t_ms, 3)
    lan_results.append(row)

lan_df = pd.DataFrame(lan_results)

# ----------------------------
# 5. Compute total write times
# ----------------------------
total_write_times = {}
for w in word_sizes:
    total_write_times[f"Total {w}-bit (ms)"] = round(lan_df[f"{w}-bit Write Time (ms)"].sum(), 3)

# Add totals as a new row in DataFrame
total_row = {"Layer": "Total", "Output Dim": "-"}
for w in word_sizes:
    total_row[f"{w}-bit Write Time (ms)"] = total_write_times[f"Total {w}-bit (ms)"]
lan_df = pd.concat([lan_df, pd.DataFrame([total_row])], ignore_index=True)

# Print totals
print("===== LanNet Total Write Times (ms) =====")
for k, v in total_write_times.items():
    print(f"{k}: {v} ms")

# Save to CSV
lan_df.to_csv("LanNet_write_times.csv", index=False)

# ----------------------------
# 6. Plot write times
# ----------------------------
layers = lan_df["Layer"].tolist()
t16 = lan_df["16-bit Write Time (ms)"].tolist()
t32 = lan_df["32-bit Write Time (ms)"].tolist()
t64 = lan_df["64-bit Write Time (ms)"].tolist()

bar_width = 0.25
x = np.arange(len(layers))

plt.figure(figsize=(12,6))
plt.bar(x - bar_width, t16, width=bar_width, color='skyblue', label='16-bit')
plt.bar(x, t32, width=bar_width, color='orange', label='32-bit')
plt.bar(x + bar_width, t64, width=bar_width, color='green', label='64-bit')

plt.xticks(x, layers, rotation=45)
plt.xlabel("Layers")
plt.ylabel("Write Time (ms)")
plt.title("LanNet Layer-wise Write Times (Dual-mode)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
