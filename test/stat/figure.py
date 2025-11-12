import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

labels = ['Category 1', 'Category 2', 'Category 3']
group1 = [20, 34, 30]
group2 = [25, 32, 34]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 6))

rects1 = ax.bar(x - width/2, group1, width, label='Group 1', color='coral')
rects2 = ax.bar(x + width/2, group2, width, label='Group 2', color='steelblue')

ax.set_ylabel('Values')
ax.set_title('Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
# plt.savefig('bar_chart_square.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()