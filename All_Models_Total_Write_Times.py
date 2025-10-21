import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Prepare summary data
# ----------------------------
data = {
    "Neural Network Model": ["LeNet", "DanNet", "AlexNet", "VGG-16", "ResNet-18"],
    "16-bit Write Time (ms)": [886.801, 1688.084, 28612.011, 461849.182, 81276.185],
    "32-bit Write Time (ms)": [443.399, 844.043, 14306.002, 230924.589, 40638.093],
    "64-bit Write Time (ms)": [221.702, 422.021, 7153.002, 115462.297, 20319.046]
}

df = pd.DataFrame(data)

# ----------------------------
# 2. Save to CSV
# ----------------------------
df.to_csv("NeuralNetwork_WriteTimes_Summary.csv", index=False)

# ----------------------------
# 3. Plot comparison (log scale)
# ----------------------------
models = df["Neural Network Model"].tolist()
t16 = df["16-bit Write Time (ms)"].tolist()
t32 = df["32-bit Write Time (ms)"].tolist()
t64 = df["64-bit Write Time (ms)"].tolist()

bar_width = 0.25
x = np.arange(len(models))

plt.figure(figsize=(10,6))
plt.bar(x - bar_width, t16, width=bar_width, color='skyblue', label='16-bit')
plt.bar(x, t32, width=bar_width, color='orange', label='32-bit')
plt.bar(x + bar_width, t64, width=bar_width, color='green', label='64-bit')

plt.xticks(x, models, rotation=45)
plt.xlabel("Neural Network Models")
plt.ylabel("Total Write Time (ms, log scale)")
plt.title("Comparison of Total Write Times for Different Word Sizes (Log Scale)")
plt.yscale("log")  # logarithmic scale for y-axis
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
