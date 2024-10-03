import os
import matplotlib.pyplot as plt
import seaborn as sns

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

dir_path1 = 'real_data/augmented_real_infill/train/'
dir_path2 = 'real_data/augmented_real_infill/test/'
labels = ['good', 'over', 'under']
colours =['#00e55e','#ff8484','#8484ff','#ff4000','#0040ff']
counts = []
for label in labels:
    counts.append(len([entry for entry in os.listdir(dir_path1+label) if os.path.isfile(os.path.join(dir_path1+label, entry))]) + len([entry for entry in os.listdir(dir_path2+label) if os.path.isfile(os.path.join(dir_path2+label, entry))]))

dir_path1 = 'real_data/augmented_real_line/train/'
dir_path2 = 'real_data/augmented_real_line/test/'
labels = ['good', 'over', 'under']
colours =['#00e55e','#ff8484','#8484ff','#ff4000','#0040ff']

for label in labels:
    counts.append(len([entry for entry in os.listdir(dir_path1+label) if os.path.isfile(os.path.join(dir_path1+label, entry))]) + len([entry for entry in os.listdir(dir_path2+label) if os.path.isfile(os.path.join(dir_path2+label, entry))]))

labels = ['infill_good', 'infill_over', 'infill_under','line_good', 'line_over', 'line_under']
print(counts)
plt.rcParams['axes.titley'] = 1.0
plt.rcParams['axes.titlepad'] = -10
plt.pie(counts, labels=labels, autopct=autopct_format(counts),colors=sns.color_palette('crest'), radius=1, pctdistance=.8, labeldistance=1.06, startangle=200)
#plt.title("Proportion of each class in real dataset")
plt.savefig("dataset_split.png",dpi=300)
plt.show