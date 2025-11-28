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
group0 = [16.96, 13.39]
group1 = [22.77, 14.8]
group2 = [27.23, 16.96]
group3 = [15, 9.82]
group4 = [12.95, 6.7]
# groups = [group1, group2, group3]
# colors = ['orange', 'green', 'purple']
# labels_group = ['CoT', 'CoVe', 'Our Approach']

x = np.arange(len(labels))
width = 0.1

fig, ax = plt.subplots(figsize=(7.5, 7.5))
x = np.array([0.0, 0.6])
n_groups = 5
offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * width
rec0 = ax.bar(x + offsets[0], group0, width, label='CoT', color='red')
rec1 = ax.bar(x + offsets[1], group1, width, label='CoVe', color='limegreen')
rec2 = ax.bar(x + offsets[2], group2, width, label='CoVe - Search Engine', color='cyan')
rec3 = ax.bar(x + offsets[3], group3, width, label='Our Approach', color='darkviolet')
rec4 = ax.bar(x + offsets[4], group4, width, label='Our Approach - mutations', color='darkorange')

ax.set_ylabel('Hallucination (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Hallucination Rate after Repair')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.45, -0.05),
    ncol=3,
    handlelength=1.2,
    columnspacing=0.9,
)

for rects in [rec0, rec1, rec2, rec3, rec4]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.savefig('recheck_hallucination.png')
# plt.savefig('recheck_hallucination.pdf')
plt.show()