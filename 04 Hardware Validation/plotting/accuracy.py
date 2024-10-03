import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  
import pandas as pd
data = []
dataset = ["real", "transfer"]
print_type = ["line", "infill"]
models = ["MobileNet","ResNet50","Xception","VGG19"]
for m in models:
    for pt in print_type:
        for ds in dataset:
            if m == "VGG19" and pt == "infill" and ds == "transfer":
                data.append({"model":f"{m} {pt}", "training_type": ds, "accuracy":0})
                continue
            d = []
            for i in range(1,6,1):
                f = open(f"accuracy/{ds}/{pt}/{m}/{i}.txt")
                f.readline()
                total = 0
                tp = 0
                for line in f.readlines():
                    total+=1
                    temp = line.split(":")
                    if temp[0] in temp[1]:
                        tp+=1
                
                data.append({"model":f"{m} {pt}", "training_type": ds, "accuracy":(tp/total)*100})
print(len(data))
data_df =  pd.DataFrame(data)
'''
data_df = pd.DataFrame({f'{models[0]} {print_type[0]} {dataset[0]}':data[0],
                        f'{models[0]} {print_type[0]} {dataset[1]}':data[1],
                        f'{models[0]} {print_type[1]} {dataset[0]}':data[2],
                        f'{models[0]} {print_type[1]} {dataset[1]}':data[3],

                        f'{models[1]} {print_type[0]} {dataset[0]}':data[4],
                        f'{models[1]} {print_type[0]} {dataset[1]}':data[5],
                        f'{models[1]} {print_type[1]} {dataset[0]}':data[6],
                        f'{models[1]} {print_type[1]} {dataset[1]}':data[7],

                        f'{models[2]} {print_type[0]} {dataset[0]}':data[8],
                        f'{models[2]} {print_type[0]} {dataset[1]}':data[9],
                        f'{models[2]} {print_type[1]} {dataset[0]}':data[10],
                        f'{models[2]} {print_type[1]} {dataset[1]}':data[11],
                        

                        f'{models[3]} {print_type[0]} {dataset[0]}':data[12],
                        f'{models[3]} {print_type[0]} {dataset[1]}':data[13],
                        f'{models[3]} {print_type[1]} {dataset[0]}':data[14],
                        f'{models[3]} {print_type[1]} {dataset[1]}':data[15],
                      })
'''

print(data_df) 
'''
data_df = data_df.melt(var_name="Model",value_name="Accuracy (%)")
print(data_df) 
'''
 
#fig = plt.figure(figsize =(10, 7))
 
# Creating plot
#plt.boxplot(data)
hatches=['','','','','','','','','oo','oo','oo','oo','oo','oo','oo','oo','','oo']
edges=['seagreen','seagreen','seagreen','seagreen','seagreen','seagreen','seagreen','seagreen','dimgrey','dimgrey','dimgrey','dimgrey','dimgrey','dimgrey','dimgrey','dimgrey','seagreen','dimgrey']
# show plot
#plt.show()
fig, ax = plt.subplots(figsize =(8, 4))
#sns.boxplot(x="Model",y="Accuracy (%)",data=data_df,palette="Paired")
bar = sns.barplot(data=data_df, x="model", y="accuracy", hue="training_type", palette=['lightgreen','darkgray'],order=["VGG19 line","VGG19 infill","MobileNet line","MobileNet infill", "ResNet50 line","ResNet50 infill","Xception line","Xception infill"])
for i,thisbar in enumerate(bar.patches):
    print(i)
    print(thisbar)
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])
    thisbar.set_edgecolor(edges[i])
plt.title("Validation Accuracy Running on Hardware")
plt.xlabel('Model') 
plt.ylabel('Accuracy (%)') 
plt.xticks(rotation=35)
plt.ylim(50, 100)
plt.yticks([50,60,70,80,90,100])
plt.yticks(ticks=[55,65,75,85,95],minor=True)
plt.grid(axis='y', which='major', linestyle='-')
plt.grid(axis='y', which='minor', color='gainsboro', linestyle='--')
ax.set_axisbelow(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Model Accuracy on Hardware')
plt.tight_layout()


plt.savefig("validation-accuracies.png",dpi=300)