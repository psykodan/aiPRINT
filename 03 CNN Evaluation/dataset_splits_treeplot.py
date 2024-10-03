import os
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

dir_path1 = 'synth_data/augmented_synth_infill/train/'
dir_path2 = 'synth_data/augmented_synth_infill/test/'
labels = ['good', 'over', 'under']
colours =['#00e55e','#ff8484','#8484ff','#ff4000','#0040ff']
counts = []
for label in labels:
    counts.append(len([entry for entry in os.listdir(dir_path1+label) if os.path.isfile(os.path.join(dir_path1+label, entry))]) + len([entry for entry in os.listdir(dir_path2+label) if os.path.isfile(os.path.join(dir_path2+label, entry))]))

dir_path1 = 'synth_data/augmented_synth_line/train/'
dir_path2 = 'synth_data/augmented_synth_line/test/'
labels = ['good', 'over', 'under']
colours =['#00e55e','#ff8484','#8484ff','#ff4000','#0040ff']

for label in labels:
    counts.append(len([entry for entry in os.listdir(dir_path1+label) if os.path.isfile(os.path.join(dir_path1+label, entry))]) + len([entry for entry in os.listdir(dir_path2+label) if os.path.isfile(os.path.join(dir_path2+label, entry))]))

labels = ['infill_good', 'infill_over', 'infill_under','line_good', 'line_over', 'line_under']
print(counts)
total = sum(counts)

i = 0
for l in labels:
    labels[i] = f"{l}\n{(counts[i]/total)*100:.2f}%\n{counts[i]}"
    i+=1

'''
plt.rcParams['axes.titley'] = 1.0
plt.rcParams['axes.titlepad'] = -10
plt.pie(counts, labels=labels, autopct=autopct_format(counts),colors=sns.color_palette('crest'), radius=1, pctdistance=.8, labeldistance=1.06, startangle=200)
#plt.title("Proportion of each class in real dataset")
plt.savefig("dataset_split.png",dpi=300)
plt.show
'''

squarify.plot(sizes=counts, label=labels,
              text_kwargs={'fontsize': 14},
              color=sns.color_palette("flare",  
                                     len(counts))) 
plt.axis("off")
plt.savefig("dataset_split_synth.png",dpi=300)