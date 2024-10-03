import matplotlib.pyplot as plt
import json
import os

models = ["vgg19","MobileNetV3","ResNet50","Xception"]
model_type = "synth"
dir= f"results-{model_type}"
for file in os.listdir(dir):
    if file.endswith(".json"):
        print(os.path.join(dir, file))
        fig, ax = plt.subplots(figsize =(5, 4))
        # Opening JSON file
        f = open(os.path.join(dir, file))

        # returns JSON object as 
        # a dictionary
        data = json.load(f)

        # summarize history for accuracy
        plt.plot(data['accuracy'],'c')
        plt.plot(data['val_accuracy'],'b',linestyle='dashdot')
        plt.plot(data['loss'],'m')
        plt.plot(data['val_loss'],'r',linestyle='dashdot')
        title = ''
        for m in models:
            if m in file:
                title = m
                if 'line' in file:
                    title +="_line"
                else:
                    title +="_infill"
        plt.title(f'{title} model accuracy/loss')
        plt.ylabel('accuracy/loss')
        plt.ylim(-0.1, 1.1)
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.xlabel('epoch')
        plt.legend(['Train-accuracy', 'Validation-accuracy','Train-loss', 'Validation-loss'], loc='center')
        #plt.show()
        plt.tight_layout()
        plt.savefig(f"acc-loss-mini-{title}-{model_type}.png",dpi=300)