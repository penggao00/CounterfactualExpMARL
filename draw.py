import numpy as np
import matplotlib.pyplot as plt

# Your provided data
observation = np.array([
    [1, 7, 0, 0, -2],
    [0, 9, 0, 0,  0],
    [0, 9, 0, 0,  0],
    [0, 9, 0, 0,  0],
    [0, 7, 0, 0, -1]
])

# Agents' positions and items (row, column, item)
agents = {
    'agent_1': [0, 0, 0],
    'agent_2': [0, 4, 2]
}

fig, ax = plt.subplots(figsize=(6,6))

# Plot the grid world
cmap = plt.cm.coolwarm
ax.imshow(observation, cmap=cmap, interpolation='none')

# Add annotations (cell values)
for (j, i), val in np.ndenumerate(observation):
    ax.text(i, j, val, ha='center', va='center', fontsize=12, color='black')

# Mark agent positions and items
for agent_name, (row, col, item) in agents.items():
    ax.scatter(col, row, marker='o', s=300, label=agent_name, edgecolors='black', linewidths=1.5)
    ax.text(col, row, agent_name, ha='center', va='center', color='white', fontsize=9)
    if item != 0:
        ax.text(col, row+0.3, f'Item: {item}', ha='center', va='center', color='yellow', fontsize=7, fontweight='bold')

# Set up grid lines
ax.set_xticks(np.arange(-0.5, observation.shape[1], 1))
ax.set_yticks(np.arange(-0.5, observation.shape[0], 1))
ax.grid(color='black', linestyle='-', linewidth=1)

# Remove axis labels for clarity
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_title('Grid World Visualization with Agents and Items', fontsize=14)
# plt.legend(loc='upper right')
plt.gca().invert_yaxis()
plt.show()