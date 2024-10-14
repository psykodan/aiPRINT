import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras

datatypes = ["synth","real","transfer"]
model_names = ["vgg19","MobileNetV3Small","Xception","ResNet50V2"]

for name in range(4):
    model_line = [tf.keras.models.load_model(f'05 Data/models/results-synth/{model_names[name]}_line.h5'),
                tf.keras.models.load_model(f'05 Data/models/results-real/{model_names[name]}_line.h5'),
                tf.keras.models.load_model(f'05 Data/models/results-transfer/{model_names[name]}_line.h5')]
    model_infill = [tf.keras.models.load_model(f'05 Data/models/results-synth/{model_names[name]}_infill.h5'),
                    tf.keras.models.load_model(f'05 Data/models/results-real/{model_names[name]}_infill.h5'),
                    tf.keras.models.load_model(f'05 Data/models/results-transfer/{model_names[name]}_infill.h5')]
    batch_size = 64
    img_h = 224
    img_w = 224
    for run in range(3):
        if run == 0:
            dataset_path_test_line = '05 Data/datasets/synth_data/augmented_synth_line/test'
            dataset_path_test_infill = '05 Data/datasets/synth_data/augmented_synth_infill/test'
            
        else:
            dataset_path_test_line = '05 Data/datasets/real_data/augmented_real_line/test'
            dataset_path_test_infill = '05 Data/datasets/real_data/augmented_real_infill/test'
            
        test_ds_line = tf.keras.utils.image_dataset_from_directory(
            dataset_path_test_line,
            label_mode="categorical",
            seed=1678,
            image_size=(img_h, img_w),
            batch_size=batch_size,
        )
        test_ds_infill = tf.keras.utils.image_dataset_from_directory(
            dataset_path_test_infill,
            label_mode="categorical",
            seed=1678,
            image_size=(img_h, img_w),
            batch_size=batch_size,
        )

    #test_ds = test_ds.skip(340)


        class_names=["+","o","-"]

        if run == 0:
            colour = sns.light_palette("cornflowerblue", as_cmap=True)
        elif run == 1:
            colour = sns.light_palette("lightgreen", as_cmap=True)
        else:
            colour = sns.light_palette("lightcoral", as_cmap=True)
        num_correct = 0

        for repeat in range(1):
            test_images = []
            test_labels = []
            #for test in range(100):
            for x,y in test_ds_line.as_numpy_iterator(): 
                if len(test_images) < 512:
                    for i in range(64):
                        image = x[i].astype("uint8")
                        test_images.append(image)
                        test_labels.append(y[i])
            labels=[]
            for num, t_img in enumerate(test_images[512*repeat:512*(repeat+1)]):  

                predictions = model_line[run].predict(np.expand_dims(t_img, axis=0), verbose = 0)
                score = tf.nn.softmax(predictions[0])
                labels.append(score)
            predicted_labels = np.argmax(labels, axis=1)
            true_labels = np.argmax(test_labels, axis=1)
            conf_matrix = confusion_matrix(true_labels, predicted_labels,labels=[1,0,2])
            print(conf_matrix)

            # Plot the confusion matrix using seaborn and matplotlib
            plt.figure(figsize=(1, 1))
            g=sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=colour, cbar=False,
                        xticklabels=class_names,
                        yticklabels=class_names,annot_kws={"size": 8})
            #plt.title(f'{m}',fontsize=10)
            #plt.xlabel('Predicted Label')
            #plt.ylabel('True Label')
            #plt.tight_layout()
            plt.savefig(f"05 Data/evaluation/confusion matrix/line-{datatypes[run]}-{model_names[name]}_CM.png",dpi=300)
            #plt.show()

        for repeat in range(1):
            test_images = []
            test_labels = []
            #for test in range(100):
            for x,y in test_ds_infill.as_numpy_iterator(): 
                if len(test_images) < 512:
                    for i in range(64):
                        image = x[i].astype("uint8")
                        test_images.append(image)
                        test_labels.append(y[i])
            labels=[]
            for num, t_img in enumerate(test_images[512*repeat:512*(repeat+1)]):  

                predictions = model_infill[run].predict(np.expand_dims(t_img, axis=0), verbose = 0)
                score = tf.nn.softmax(predictions[0])
                labels.append(score)
            predicted_labels = np.argmax(labels, axis=1)
            true_labels = np.argmax(test_labels, axis=1)
            conf_matrix = confusion_matrix(true_labels, predicted_labels,labels=[1,0,2])
            print(conf_matrix)

            # Plot the confusion matrix using seaborn and matplotlib
            plt.figure(figsize=(1, 1))
            g=sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=colour, cbar=False,
                        xticklabels=class_names,
                        yticklabels=class_names,annot_kws={"size": 8})
            #plt.title(f'{m}',fontsize=10)
            #plt.xlabel('Predicted Label')
            #plt.ylabel('True Label')
            #plt.tight_layout()
            plt.savefig(f"05 Data/evaluation/confusion matrix/infill-{datatypes[run]}-{model_names[name]}_CM.png",dpi=300)
            #plt.show()
                



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