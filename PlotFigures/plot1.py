import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib import rcParams

# Set global font to Times New Roman
rcParams['font.family'] = 'Times New Roman'

# Settings
labels = ['LLM', 'StaticRAG', 'DynamicRAG']
x = np.arange(len(labels))
width = 0.4  # bar width

# Data
openai_sc = [178, 182, 202]
openai_fm = [101, 101, 120]
codex_sc = [208, 213, 215]
codex_fm = [110, 134, 142]
deepseek_sc = [210, 210, 212]
deepseek_fm = [174, 170, 172]

# Define colors
sc_color = 'skyblue'
fm_color = 'salmon'

# Start Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

models = ['GPT-4o-mini', 'CodeX', 'DeepSeek']
all_sc = [openai_sc, codex_sc, deepseek_sc]
all_fm = [openai_fm, codex_fm, deepseek_fm]

for ax, sc, fm, model in zip(axes, all_sc, all_fm, models):
    bars_sc = ax.bar(x - width/2, sc, width, color=sc_color)
    bars_fm = ax.bar(x + width/2, fm, width, color=fm_color)

    ax.tick_params(axis='y', labelsize=14)

    ax.set_title(model, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=14)
    # ax.set_yticklabels(fontsize=14)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 250)

    # Annotate SC and FM
    for i in range(len(labels)):
        # For SC
        if i == 0:
            ax.annotate(f'{sc[i]}',
                        xy=(x[i] - width/2, sc[i]),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)
        else:
            improv = (sc[i] - sc[0]) / sc[0] * 100
            ax.annotate(f'{improv:+.1f}%',
                        xy=(x[i] - width/2, sc[i]),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

        # For FM
        if i == 0:
            ax.annotate(f'{fm[i]}',
                        xy=(x[i] + width*2 / 3, fm[i]),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)
        else:
            improv = (fm[i] - fm[0]) / fm[0] * 100
            ax.annotate(f'{improv:+.1f}%',
                        xy=(x[i] + width * 2 / 3, fm[i]),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    # Add dashed line at 229
    ax.axhline(y=229, linestyle='--', color='red', linewidth=1)
    ax.text(x[-1] + 0.3, 229, '229', ha='center', va='bottom', color='red', fontsize=14)

# Global figure labels
# fig.text(0.5, 0.02, 'Setting', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Number of Assertions', va='center', rotation='vertical', fontsize=14)

# Add a proper legend
legend_elements = [
    Patch(facecolor=sc_color, edgecolor='black', label='Syntax Correctness (SC)'),
    Patch(facecolor=fm_color, edgecolor='black', label='Functionality Match (FM)')
]
fig.legend(handles=legend_elements, loc='upper center',
           ncol=2, frameon=True, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
