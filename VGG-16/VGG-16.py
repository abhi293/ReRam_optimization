import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. VGG-16 Architecture (for reference)
# ----------------------------
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# ----------------------------
# 2. Dual-mode write time function
# ----------------------------
def dual_mode_write_time(num_bits, word_size_bits, slow_ns=20000, fast_ns=500, access_ns=5, weight_slow=0.75, weight_fast=0.25):
    num_words = num_bits / word_size_bits
    t_word_s = weight_slow * (access_ns*1e-9 + slow_ns*1e-9) + weight_fast * (access_ns*1e-9 + fast_ns*1e-9)
    t_total_ms = num_words * t_word_s * 1000  # convert s to ms
    return t_total_ms

# ----------------------------
# 3. VGG-16 Layer info (Output dimensions)
# ----------------------------
layer_info = [
    {"name": "Input", "H": 224, "W": 224, "D": 3},
    # Block 1
    {"name": "Conv1_1", "H":224, "W":224, "D":64},
    {"name": "Conv1_2", "H":224, "W":224, "D":64},
    {"name": "Pool1", "H":112, "W":112, "D":64},
    # Block 2
    {"name": "Conv2_1", "H":112, "W":112, "D":128},
    {"name": "Conv2_2", "H":112, "W":112, "D":128},
    {"name": "Pool2", "H":56, "W":56, "D":128},
    # Block 3
    {"name": "Conv3_1", "H":56, "W":56, "D":256},
    {"name": "Conv3_2", "H":56, "W":56, "D":256},
    {"name": "Conv3_3", "H":56, "W":56, "D":256},
    {"name": "Pool3", "H":28, "W":28, "D":256},
    # Block 4
    {"name": "Conv4_1", "H":28, "W":28, "D":512},
    {"name": "Conv4_2", "H":28, "W":28, "D":512},
    {"name": "Conv4_3", "H":28, "W":28, "D":512},
    {"name": "Pool4", "H":14, "W":14, "D":512},
    # Block 5
    {"name": "Conv5_1", "H":14, "W":14, "D":512},
    {"name": "Conv5_2", "H":14, "W":14, "D":512},
    {"name": "Conv5_3", "H":14, "W":14, "D":512},
    {"name": "Pool5", "H":7, "W":7, "D":512},
    # FC layers
    {"name": "Flatten", "H":1, "W":1, "D":512*7*7},
    {"name": "FC1", "H":1, "W":1, "D":4096},
    {"name": "FC2", "H":1, "W":1, "D":4096},
    {"name": "FC3", "H":1, "W":1, "D":1000},
]

# ----------------------------
# 4. Compute write times
# ----------------------------
word_sizes = [16, 32, 64]  # in bits
results = []

for layer in layer_info:
    num_bits = layer["H"] * layer["W"] * layer["D"] * 32  # base 32-bit elements
    row = {"Layer": layer["name"], "Output Dim": f"{layer['H']}x{layer['W']}x{layer['D']}"}
    for w in word_sizes:
        t_ms = dual_mode_write_time(num_bits=num_bits, word_size_bits=w)
        row[f"{w}-bit Write Time (ms)"] = round(t_ms,3)
    results.append(row)

df = pd.DataFrame(results)

# ----------------------------
# 5. Compute total write times
# ----------------------------
total_times = {w: df[f"{w}-bit Write Time (ms)"].sum() for w in word_sizes}

print("\n===== VGG-16 Model Total Write Times =====")
for w in word_sizes:
    print(f"Total {w}-bit (ms): {round(total_times[w],3)} ms")

# Append total row to CSV
total_row = {
    "Layer": "TOTAL",
    "Output Dim": "-",
    "16-bit Write Time (ms)": round(total_times[16],3),
    "32-bit Write Time (ms)": round(total_times[32],3),
    "64-bit Write Time (ms)": round(total_times[64],3),
}
df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# Save CSV
df.to_csv("VGG16_write_times.csv", index=False)

# ----------------------------
# 6. Plot write times
# ----------------------------
layers = df["Layer"].tolist()
t16 = df["16-bit Write Time (ms)"].tolist()
t32 = df["32-bit Write Time (ms)"].tolist()
t64 = df["64-bit Write Time (ms)"].tolist()

bar_width = 0.25
x = np.arange(len(layers))

plt.figure(figsize=(16,6))
plt.bar(x - bar_width, t16, width=bar_width, color='skyblue', label='16-bit')
plt.bar(x, t32, width=bar_width, color='orange', label='32-bit')
plt.bar(x + bar_width, t64, width=bar_width, color='green', label='64-bit')

plt.xticks(x, layers, rotation=90)
plt.xlabel("Layers")
plt.ylabel("Write Time (ms)")
plt.title("VGG-16 Layer-wise Write Times (Dual-mode)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
