# Re-adjusting the legend position to avoid overlapping the last bar's labels

import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
models = ['LR', 'NB', 'RF']
datasets = ['Netejat 1', 'Netejat 2', 'Netejat 3']
accuracies = [
    [82.20, 79.96, 82.14],  # LR
    [79.96, 78.33, 79.97],  # NB
    [70.76, 69.53, 69.99],  # RF
]

# Plotting
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, (dataset, color) in enumerate(zip(datasets, ['blue', 'orange', 'green'])):
    ax.bar(x + i * width, [accuracy[i] for accuracy in accuracies], width, label=dataset, color=color)

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Exactitud (%)')
ax.set_title("Comparació de l'exactitud amb els diferents models")
ax.set_xticks(x + width)
ax.set_xticklabels(models)
ax.legend(title='Datasets', loc='upper left', bbox_to_anchor=(1, 1))

# Add data labels
for i in range(len(models)):
    for j in range(len(datasets)):
        ax.text(x[i] + j * width, accuracies[i][j], f"{accuracies[i][j]:.2f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
