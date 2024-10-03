from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

models = ["vgg19","MobileNetV3","ResNet50","Xception"]
#models = ["vgg19_line","vgg19_infill","MobileNetV3_line","MobileNetV3_infill","ResNet50_line","ResNet50_infill","Xception_line","Xception_infill"]
#res = [[924, 875], [938, 874], [919, 877], [934, 873]]
synth_res = [[92.4, 93.8, 91.9, 93.4],[87.5, 87.4, 87.7, 87.3]]
real_res = [[98.4,98.7,98.5,98.6,],[95.8, 95.7,96.8,96.3]]
res = [[98.3,98.3,97.9,97.9],[56.1,96.1,95.9,95.8]]
# Figure Size
fig, ax = plt.subplots(figsize =(8, 4))
barWidth = 0.1

# Set position of bar on X axis 
br3 = np.arange(4) 
br2 = [x - barWidth for x in br3]
br1 = [x - barWidth for x in br2]
 
 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4] 
br6 = [x + barWidth for x in br5] 

# Horizontal Bar Plot
plt.bar(br1, synth_res[0], edgecolor ='coral', color='white',width = barWidth, label ='synth_line', hatch='..') 
plt.bar(br2, real_res[0], edgecolor ='lightgreen',color='white', width = barWidth, label ='real_line',hatch='') 
plt.bar(br3, res[0], edgecolor ='darkgray', color='white',width = barWidth, label ='transfer_line',hatch='oo') 
plt.bar(br4, synth_res[1], color ='coral',edgecolor ='orangered', width = barWidth, label ='synth_infill',hatch='..') 
plt.bar(br5, real_res[1], color ='lightgreen', edgecolor ='seagreen',width = barWidth, label ='real_infill',hatch='') 
plt.bar(br6, res[1], color ='darkgray', edgecolor ='dimgrey',width = barWidth, label ='transfer_infill',hatch='oo')
 
# Adding Xticks 
plt.xlabel('Model') 
plt.ylabel('Accuracy (%)') 
plt.xticks([r + barWidth/2 for r in range(4)], 
        models)
plt.ylim(50, 100)
plt.yticks([50,60,70,80,90,100])
plt.yticks(ticks=[55,65,75,85,95],minor=True)
plt.grid(axis='y', which='major', linestyle='-')
plt.grid(axis='y', which='minor', color='gainsboro', linestyle='--')
ax.set_axisbelow(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Model Accuracy on Holdout Data')
plt.tight_layout()
# Show Plot
plt.savefig("accuracies.png",dpi=300)
