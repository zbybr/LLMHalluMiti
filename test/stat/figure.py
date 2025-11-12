import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0

labels = ['gpt-4o', 'gpt-5']
group1 = [20, 34]
group2 = [25, 32]
group3 = [25, 32]
# groups = [group1, group2, group3]
# colors = ['orange', 'green', 'purple']
# labels_group = ['CoT', 'CoVe', 'Our Approach']

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(6, 6))

rec1 = ax.bar(x - width, group1, width, label='CoT', color='orange')
rec2 = ax.bar(x, group2, width, label='CoVe', color='green')
rec3 = ax.bar(x + width, group3, width, label='Our Approach', color='purple')

ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Recheck Hallucination Rate')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    handlelength=1.2,
    columnspacing=1.5,
)

for rects in [rec1, rec2, rec3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# plt.savefig('bar_chart_square.png')
plt.show()