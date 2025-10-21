import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Load ResNet-18 model
# ----------------------------
resnet18 = models.resnet18(pretrained=False)  # using untrained model
resnet18.eval()  # set to eval mode

# ----------------------------
# 2. Hook function to record layer outputs
# ----------------------------
layer_outputs = {}

def hook_fn(module, input, output):
    layer_name = module.__class__.__name__ + "_" + str(id(module))  # unique name
    layer_outputs[layer_name] = output.shape

# Register hooks on all Conv, Linear, and pooling layers
for name, module in resnet18.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        module.register_forward_hook(hook_fn)

# ----------------------------
# 3. Pass a sample input through the model
# ----------------------------
sample_input = torch.randn(1, 3, 224, 224)  # batch size = 1
with torch.no_grad():
    _ = resnet18(sample_input)

# ----------------------------
# 4. Prepare layer info from real outputs
# ----------------------------
layer_info = []
for k, shape in layer_outputs.items():
    # For Conv2d/Pool2d: shape = [batch, channels, H, W]; for Linear: shape = [batch, features]
    if len(shape) == 4:
        H, W, D = shape[2], shape[3], shape[1]
    elif len(shape) == 2:
        H, W, D = 1, 1, shape[1]
    else:
        continue
    layer_info.append({"name": k, "H": H, "W": W, "D": D})

# ----------------------------
# 5. Dual-mode write time function
# ----------------------------
def dual_mode_write_time(num_bits, word_size_bits, slow_ns=20000, fast_ns=500, access_ns=5, weight_slow=0.75, weight_fast=0.25):
    num_words = num_bits / word_size_bits
    t_word_s = weight_slow * (access_ns*1e-9 + slow_ns*1e-9) + weight_fast * (access_ns*1e-9 + fast_ns*1e-9)
    t_total_ms = num_words * t_word_s * 1000
    return t_total_ms

# ----------------------------
# 6. Compute write times per layer
# ----------------------------
word_sizes = [16, 32, 64]
results = []

for layer in layer_info:
    num_bits = layer["H"] * layer["W"] * layer["D"] * 32  # base 32-bit
    row = {"Layer": layer["name"], "Output Dim": f"{layer['H']}x{layer['W']}x{layer['D']}"}
    for w in word_sizes:
        t_ms = dual_mode_write_time(num_bits=num_bits, word_size_bits=w)
        row[f"{w}-bit Write Time (ms)"] = round(t_ms, 3)
    results.append(row)

df = pd.DataFrame(results)

# ----------------------------
# 7. Compute total write times
# ----------------------------
total_times = {w: df[f"{w}-bit Write Time (ms)"].sum() for w in word_sizes}

print("\n===== ResNet-18 Model Total Write Times (Real Experiment) =====")
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
df.to_csv("ResNet18_real_write_times.csv", index=False)

# ----------------------------
# 8. Plot write times
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
plt.title("ResNet-18 Layer-wise Write Times (Real Experiment, Dual-mode)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
