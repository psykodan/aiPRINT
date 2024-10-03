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
                data.append({"model":f"{m} {ds}", "dps":0, "tpd":0})
                continue
            f = open(f"speed/{ds}/{pt}/{m}.txt")
            for line in f.readlines():
                data.append({"model":f"{m} {ds}", "Detections per second":int(line.split(",")[-2].split(":")[-1])/100, "Time per prediction (ms)":float(line.split(",")[-1].split(":")[-1])*1000})
            

data_df = pd.DataFrame(data)
print(data_df) 
#data_df = data_df.melt(var_name="Model",value_name="Detections / second")
#print(data_df) 
 
#fig = plt.figure(figsize =(10, 7))
 
# Creating plot
#plt.boxplot(data)
 
# show plot
#plt.show()
plt.figure(figsize=(9, 6))
sns.scatterplot(data=data_df, x="Time per prediction (ms)", y="Detections per second", hue="model", style="model", palette="Set1")
plt.title("Detections per Second Running on Hardware")
plt.ylim(5, 22.5)
plt.xlim(20, 120)

plt.tight_layout()


plt.savefig("validation-dps.png",dpi=300)