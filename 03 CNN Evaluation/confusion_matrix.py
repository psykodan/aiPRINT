import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras
run = "synth"


models ={"vgg19_line" : tf.keras.models.load_model('vgg19_line.h5'),
		 
         "MobileNetV3_line" : tf.keras.models.load_model('MobileNetV3Small_line.h5'),
		 
         "ResNet50_line" : tf.keras.models.load_model('ResNet50V2_line.h5'),
		
         "Xception_line" : tf.keras.models.load_model('Xception_line.h5')}
class_names=["good","over","under"]
dataset_path_test = 'real_data/augmented_real_line/test/'
test_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path_test,
    label_mode="categorical",
    seed=1337,
    image_size=(224, 224),
    batch_size=64)

#test_ds = test_ds.skip(340)


class_names=["over","good","under"]
results = []
for m in models:
    num_correct = 0

    for repeat in range(1):
        test_images = []
        test_labels = []
        #for test in range(100):
        for x,y in test_ds.as_numpy_iterator(): 
            if len(test_images) < 512:
                for i in range(64):
                    image = x[i].astype("uint8")
                    test_images.append(image)
                    test_labels.append(y[i])
        labels=[]
        for num, t_img in enumerate(test_images[512*repeat:512*(repeat+1)]):  

            predictions = models[m].predict(np.expand_dims(t_img, axis=0), verbose = 0)
            score = tf.nn.softmax(predictions[0])
            labels.append(score)
        predicted_labels = np.argmax(labels, axis=1)
        true_labels = np.argmax(test_labels, axis=1)
        conf_matrix = confusion_matrix(true_labels, predicted_labels,labels=[1,0,2])
        print(conf_matrix)

        # Plot the confusion matrix using seaborn and matplotlib
        plt.figure(figsize=(2, 2))
        g=sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'{m}',fontsize=10)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        #plt.tight_layout()
        #plt.savefig(f"real-{m}_CM.png",dpi=300)
        plt.show()
            



'''
# Extract true labels from the dataset
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
true_labels=np.argmax(true_labels, axis=1)

for m in models:
    # Make predictions using the model
    predictions = models[m].predict(test_ds)
    predicted_labels = []
    for p in predictions:
        predicted_labels.append(tf.nn.softmax(p))
    #predicted_labels = np.argmax(predictions, axis=1)
    count = 0
    for l in range(len(true_labels)):
        #print(f"{true_labels[l]} : {predicted_labels[l]}")
        if true_labels[l] == np.argmax(predicted_labels[l]):
            count +=1
    print(count/len(true_labels))



    # Create a confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True,  cmap='Blues', cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix {m}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
'''