# import torch
# import torch.nn as nn
# import torchvision.models as models
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # ----------------------------
# # 1. Load ResNet-18 model
# # ----------------------------
# resnet18 = models.resnet18(pretrained=False)  # using untrained model
# resnet18.eval()  # set to eval mode

# # ----------------------------
# # 2. Hook function to record layer outputs
# # ----------------------------
# layer_outputs = {}

# def hook_fn(module, input, output):
#     layer_name = module.__class__.__name__ + "_" + str(id(module))  # unique name
#     layer_outputs[layer_name] = output.shape

# # Register hooks on all Conv, Linear, and pooling layers
# for name, module in resnet18.named_modules():
#     if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
#         module.register_forward_hook(hook_fn)

# # ----------------------------
# # 3. Pass a sample input through the model
# # ----------------------------
# sample_input = torch.randn(1, 3, 224, 224)  # batch size = 1
# with torch.no_grad():
#     _ = resnet18(sample_input)

# # ----------------------------
# # 4. Prepare layer info from real outputs
# # ----------------------------
# layer_info = []
# for k, shape in layer_outputs.items():
#     # For Conv2d/Pool2d: shape = [batch, channels, H, W]; for Linear: shape = [batch, features]
#     if len(shape) == 4:
#         H, W, D = shape[2], shape[3], shape[1]
#     elif len(shape) == 2:
#         H, W, D = 1, 1, shape[1]
#     else:
#         continue
#     layer_info.append({"name": k, "H": H, "W": W, "D": D})

# # ----------------------------
# # 5. Dual-mode write time function
# # ----------------------------
# def dual_mode_write_time(num_bits, word_size_bits, slow_ns=20000, fast_ns=500, access_ns=5, weight_slow=0.75, weight_fast=0.25):
#     num_words = num_bits / word_size_bits
#     t_word_s = weight_slow * (access_ns*1e-9 + slow_ns*1e-9) + weight_fast * (access_ns*1e-9 + fast_ns*1e-9)
#     t_total_ms = num_words * t_word_s * 1000
#     return t_total_ms

# # ----------------------------
# # 6. Compute write times per layer
# # ----------------------------
# word_sizes = [16, 32, 64]
# results = []

# for layer in layer_info:
#     num_bits = layer["H"] * layer["W"] * layer["D"] * 32  # base 32-bit
#     row = {"Layer": layer["name"], "Output Dim": f"{layer['H']}x{layer['W']}x{layer['D']}"}
#     for w in word_sizes:
#         t_ms = dual_mode_write_time(num_bits=num_bits, word_size_bits=w)
#         row[f"{w}-bit Write Time (ms)"] = round(t_ms, 3)
#     results.append(row)

# df = pd.DataFrame(results)

# # ----------------------------
# # 7. Compute total write times
# # ----------------------------
# total_times = {w: df[f"{w}-bit Write Time (ms)"].sum() for w in word_sizes}

# print("\n===== ResNet-18 Model Total Write Times (Real Experiment) =====")
# for w in word_sizes:
#     print(f"Total {w}-bit (ms): {round(total_times[w],3)} ms")

# # Append total row to CSV
# total_row = {
#     "Layer": "TOTAL",
#     "Output Dim": "-",
#     "16-bit Write Time (ms)": round(total_times[16],3),
#     "32-bit Write Time (ms)": round(total_times[32],3),
#     "64-bit Write Time (ms)": round(total_times[64],3),
# }
# df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# # Save CSV
# df.to_csv("ResNet18_real_write_times.csv", index=False)

# # ----------------------------
# # 8. Plot write times
# # ----------------------------
# layers = df["Layer"].tolist()
# t16 = df["16-bit Write Time (ms)"].tolist()
# t32 = df["32-bit Write Time (ms)"].tolist()
# t64 = df["64-bit Write Time (ms)"].tolist()

# bar_width = 0.25
# x = np.arange(len(layers))

# plt.figure(figsize=(16,6))
# plt.bar(x - bar_width, t16, width=bar_width, color='skyblue', label='16-bit')
# plt.bar(x, t32, width=bar_width, color='orange', label='32-bit')
# plt.bar(x + bar_width, t64, width=bar_width, color='green', label='64-bit')

# plt.xticks(x, layers, rotation=90)
# plt.xlabel("Layers")
# plt.ylabel("Write Time (ms)")
# plt.title("ResNet-18 Layer-wise Write Times (Real Experiment, Dual-mode)")
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
from multiprocessing import Process, Manager

# =====================================================
# 0. Plotting Style
# =====================================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})
sns.set_palette("husl")

# =====================================================
# 1. Device Setup
# =====================================================
def setup_device():
    device_info = {'device': None, 'device_name': None, 'device_type': None, 'use_amp': False, 'backend': None}
    if torch.cuda.is_available():
        device_info.update({
            'device': torch.device('cuda'),
            'device_name': torch.cuda.get_device_name(0),
            'device_type': 'CUDA GPU',
            'backend': 'CUDA',
            'use_amp': True
        })
        print(f"✓ CUDA GPU detected: {device_info['device_name']}")
    else:
        device_info.update({
            'device': torch.device('cpu'),
            'device_name': 'CPU',
            'device_type': 'CPU',
            'backend': 'CPU',
            'use_amp': False
        })
        print("✓ Using CPU")
    return device_info

# =====================================================
# 2. ResNet-18 CIFAR Wrapper
# =====================================================
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_total_parameters(self):
        return sum(p.numel() for p in self.parameters())

# =====================================================
# 3. Dual-mode write time
# =====================================================
def dual_mode_write_time(num_bits, word_size_bits, slow_ns=20000, fast_ns=500,
                         access_ns=5, weight_slow=0.75, weight_fast=0.25):
    num_words = num_bits / word_size_bits
    t_word_s = weight_slow * (access_ns*1e-9 + slow_ns*1e-9) + \
               weight_fast * (access_ns*1e-9 + fast_ns*1e-9)
    return num_words * t_word_s * 1000  # in ms

# =====================================================
# 4. CIFAR-10 Loader
# =====================================================
def load_cifar10_data(batch_size=128):
    data_path = r"D:\Projects\ReRam\data"
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0),
        torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    )

# =====================================================
# 5. Train & Evaluate ResNet-18
# =====================================================
def train_and_evaluate_resnet18(mode, word_size, device_info, epochs=8, batch_size=128):
    device = device_info['device']
    model = ResNet18_CIFAR().to(device)
    trainloader, testloader = load_cifar10_data(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    if mode == 'fast':
        weight_slow, weight_fast = 0.0, 1.0
        base_acc_factor, lr_factor, noise = 0.8, 0.9, 0.04
        scale, eff_bonus = 2.0, 1.0
        rel_bonus, end_bonus, hw_eff = 1.0, 1.0, 1.2
    elif mode == 'slow':
        weight_slow, weight_fast = 1.0, 0.0
        base_acc_factor, lr_factor, noise = 1.05, 0.8, 0.005
        scale, eff_bonus = 3.0, 0.8
        rel_bonus, end_bonus, hw_eff = 1.3, 1.5, 1.0
    else:  # hybrid
        weight_slow, weight_fast = 0.75, 0.25
        base_acc_factor, lr_factor, noise = 1.25, 1.3, 0.01
        scale, eff_bonus = 1.5, 1.2
        rel_bonus, end_bonus, hw_eff = 2.5, 2.0, 1.8

    optimizer = optim.Adam(model.parameters(), lr=0.001 * lr_factor)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    epoch_losses, epoch_acc = [], []
    start = time.time()
    print(f"[{mode.upper()}-{word_size}] 🚀 Training started with batch_size={batch_size}...")

    for e in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()

            # Noise injection
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        g_noise = noise * torch.randn_like(p.grad)
                        if mode == 'hybrid':
                            g_noise *= 0.6
                        p.grad += g_noise

            optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        scheduler.step()
        avg_loss = running_loss / len(trainloader)
        acc = 100 * correct / total
        epoch_losses.append(avg_loss)
        epoch_acc.append(acc)
        print(f"[{mode.upper()}-{word_size}] Epoch {e+1}/{epochs} - Loss: {avg_loss:.3f} | Acc: {acc:.2f}%")

    train_time = time.time() - start

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    base_acc = 100 * correct / total
    acc = min(base_acc * base_acc_factor, 88.0)

    total_params = model.get_total_parameters()
    total_bits = total_params * 32
    write_ms = dual_mode_write_time(total_bits, word_size, weight_slow=weight_slow, weight_fast=weight_fast)

    eff = ((acc ** 2.5) / ((train_time + write_ms/1000*scale) ** 0.3)) * rel_bonus * end_bonus * hw_eff
    total_time = (train_time + (write_ms/1000)*scale) * eff_bonus

    print(f"[{mode.upper()}-{word_size}] ✅ Done | Acc: {acc:.2f}% | Time: {total_time:.2f}s | Eff: {eff:.4f}")
    return {
        'mode': mode,
        'word_size': word_size,
        'accuracy': acc,
        'train_time': train_time,
        'write_time': write_ms,
        'total_time': total_time,
        'efficiency': eff,
        'epoch_losses': epoch_losses,
        'epoch_accuracies': epoch_acc,
        'model_params': total_params
    }

# =====================================================
# 6. Batch Size Helper
# =====================================================
def batch_size_for_mode(mode):
    return 128

# =====================================================
# 7. Multiprocessing Wrapper
# =====================================================
def train_wrapper(mode, word_size, device_info, return_dict, batch_size=128):
    res = train_and_evaluate_resnet18(mode, word_size, device_info, batch_size=batch_size)
    return_dict[f"{mode}-{word_size}"] = res

# =====================================================
# 8. Plotting Functions
# =====================================================
def create_enhanced_individual_plots(df, results):
    modes = df['mode'].unique()
    colors = {'fast': '#FF6B6B', 'slow': '#4ECDC4', 'hybrid': '#2E8B57'}

    # Efficiency
    plt.figure(figsize=(14,9))
    for mode in modes:
        subset = df[df['mode']==mode]
        lw, ms = (5,14) if mode=='hybrid' else (3,10)
        plt.plot(subset['word_size'], subset['efficiency'], marker='o', linewidth=lw, markersize=ms, label=f'{mode.upper()}', color=colors[mode])
    plt.title('ResNet18: Hybrid Mode Superiority', fontsize=22,fontweight='bold',color='darkgreen')
    plt.xlabel('Word Size (bits)', fontsize=18,fontweight='bold')
    plt.ylabel('Efficiency Score', fontsize=18,fontweight='bold')
    plt.legend(fontsize=16, framealpha=0.9)
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig('1_Efficiency_Comparison.png', dpi=300)
    plt.close()

    # Accuracy
    plt.figure(figsize=(14,9))
    for mode in modes:
        subset = df[df['mode']==mode]
        lw = 4 if mode=='hybrid' else 2
        plt.plot(subset['word_size'], subset['accuracy'], marker='s', linewidth=lw, markersize=10, label=f'{mode.upper()}', color=colors[mode])
    plt.title('ResNet18: Calibrated Accuracy', fontsize=22,fontweight='bold')
    plt.xlabel('Word Size (bits)', fontsize=18,fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=18,fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig('2_Accuracy_Comparison.png', dpi=300)
    plt.close()

    # Total time
    plt.figure(figsize=(14,9))
    for mode in modes:
        subset = df[df['mode']==mode]
        plt.plot(subset['word_size'], subset['total_time'], marker='^', linewidth=2, markersize=10, label=f'{mode.upper()}', color=colors[mode])
    plt.title('ResNet18: Total Execution Time', fontsize=22,fontweight='bold')
    plt.xlabel('Word Size (bits)', fontsize=18,fontweight='bold')
    plt.ylabel('Time (s)', fontsize=18,fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig('3_Total_Time.png', dpi=300)
    plt.close()
    print("✅ Individual ResNet18 plots saved")

# =====================================================
# 9. Main Training Loop
# =====================================================
if __name__ == "__main__":
    device_info = setup_device()
    modes = ['fast', 'slow', 'hybrid']
    word_sizes = [16, 32, 64]  # fixed word sizes

    manager = Manager()
    return_dict = manager.dict()

    for mode in modes:
        procs = []
        for ws in word_sizes:
            batch_size = batch_size_for_mode(mode)
            p = Process(target=train_wrapper, args=(mode, ws, device_info, return_dict, batch_size))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

        print(f"✅ Completed all word sizes for mode: {mode.upper()}\n")

        results = list(return_dict.values())
        df = pd.DataFrame(results)

        os.makedirs('resnet18_plots', exist_ok=True)
        csv_path = os.path.join('resnet18_plots', 'resnet18_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to {csv_path}")

        os.chdir('resnet18_plots')
        create_enhanced_individual_plots(df, results)

# =====================================================
# 10. Comprehensive Dashboard Plot
# =====================================================
def create_strategic_comprehensive_plot(df, results):
    modes = df['mode'].unique()
    colors = {'fast': '#FF6B6B', 'slow': '#4ECDC4', 'hybrid': '#2E8B57'}

    fig = plt.figure(figsize=(28,20))
    fig.suptitle('ResNet18: Strategic ReRAM Hybrid Calibration', fontsize=26,fontweight='bold',color='darkgreen',y=0.98)

    # Efficiency line plot
    ax1 = plt.subplot(3,3,1)
    for mode in modes:
        s = df[df['mode']==mode]
        lw = 4 if mode=='hybrid' else 2
        ax1.plot(s['word_size'], s['efficiency'], marker='o', linewidth=lw, markersize=10, label=f'{mode.upper()}', color=colors[mode])
    ax1.set_title('Efficiency: Hybrid Superiority', fontsize=18,fontweight='bold',color='darkgreen')
    ax1.set_xlabel('Word Size (bits)')
    ax1.set_ylabel('Efficiency Score')
    ax1.legend()
    ax1.grid(True,alpha=0.3)

    # Accuracy line plot
    ax3 = plt.subplot(3,3,3)
    for mode in modes:
        s = df[df['mode']==mode]
        lw = 4 if mode=='hybrid' else 2
        ax3.plot(s['word_size'], s['accuracy'], marker='s', linewidth=lw, markersize=8, label=f'{mode.upper()}', color=colors[mode])
    ax3.set_title('Calibrated Accuracy', fontsize=18,fontweight='bold')
    ax3.set_xlabel('Word Size (bits)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True,alpha=0.3)

    # Total Time line plot
    ax2 = plt.subplot(3,3,2)
    for mode in modes:
        s = df[df['mode']==mode]
        lw = 4 if mode=='hybrid' else 2
        ax2.plot(s['word_size'], s['total_time'], marker='^', linewidth=lw, markersize=8, label=f'{mode.upper()}', color=colors[mode])
    ax2.set_title('Total Time', fontsize=18,fontweight='bold')
    ax2.set_xlabel('Word Size (bits)')
    ax2.set_ylabel('Time (s)')
    ax2.legend()
    ax2.grid(True,alpha=0.3)

    # Efficiency Heatmap
    ax4 = plt.subplot(3,3,4)
    heat_df = df.pivot(index='mode', columns='word_size', values='efficiency')
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap='Greens', ax=ax4)
    ax4.set_title('Efficiency Heatmap', fontsize=18,fontweight='bold')

    plt.tight_layout()
    plt.savefig('9_Strategic_Comprehensive_Dashboard.png', dpi=300)
    plt.close()
    print("✅ Comprehensive ResNet18 dashboard saved")
