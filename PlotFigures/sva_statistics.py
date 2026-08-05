import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "serif"]
rcParams["mathtext.default"] = "regular"  # keep math text consistent with serif font
rcParams["font.size"]   = 14


BIN_WIDTH = 2

num_ops_per_assertion = []
num_signals_per_assertion = []
ops = set()
for folder in os.listdir("Evaluation/Dataset"):
    folder_path = os.path.join("Evaluation/Dataset/",folder)
    if os.path.isdir(folder_path):
        filepath = os.path.join(folder_path,"explanation.json")
        with open(filepath,"r") as file:
            assertions = json.load(file)
        for assertion in assertions:
            if "Assert" in assertion:
                num_ops_per_assertion.append(len(assertions[assertion]["Logical Operators"]))
                num_signals_per_assertion.append(len(assertions[assertion]["Signals"]))
                ops.update(assertions[assertion]["Logical Operators"])

# print(f"{len(ops)} operators: {ops}")
num_ops_per_assertion = np.array(num_ops_per_assertion)
num_signals_per_assertion = np.array(num_signals_per_assertion)
num_ops_mean = num_ops_per_assertion.mean()  # mean number of operators
num_signals_mean = num_signals_per_assertion.mean()  # mean number of signals

# ── count how many assertions have 0, 1, 2 … N operators ──────────────
max_ops = max(int(num_ops_per_assertion.max()),int(num_signals_per_assertion.max()))           # biggest count seen
freq = np.zeros(max_ops, dtype=int)              # 0-initialised array
for n in num_ops_per_assertion:
    freq[int(n-1)] += 1

freq_signals = np.zeros(max_ops, dtype=int)           # 0-initialised array for signals
for n in num_signals_per_assertion:
    freq_signals[int(n-1)] += 1

x = np.arange(len(freq)) + 1                             # 0, 1, 2, … max_ops
x_signals = np.arange(len(freq_signals)) + 1            # 0, 1, 2, … max_ops for signals

# ── plot ───────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))

plt.bar(x, freq, width=0.8, color="#1f77b4", alpha=0.6, label="#Operators")
plt.bar(x_signals, freq_signals, width=0.8, color="#ff7f0e", alpha=0.6, label="#Signals")
# plt.xlim(left=-0.6)
plt.xticks(x)                                        # tick at every integer
# plt.xlabel("Number of SVA Operators per SVA")
plt.ylabel("Number of SVAs")
plt.legend(prop={'size': 14})
plt.tight_layout()
# plt.show()
plt.savefig("PlotFigures/op_signal_statistics.png", dpi=300)