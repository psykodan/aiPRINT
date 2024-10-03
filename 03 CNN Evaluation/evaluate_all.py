import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random


models ={"vgg19_line" : tf.keras.models.load_model('results-transfer/vgg19_line.h5'),
		 "vgg19_infill" : tf.keras.models.load_model('results-transfer/vgg19_infill.h5'),
         "MobileNetV3_line" : tf.keras.models.load_model('results-transfer/MobileNetV3Small_line.h5'),
		 "MobileNetV3_infill" : tf.keras.models.load_model('results-transfer/MobileNetV3Small_infill.h5'),
         "ResNet50_line" : tf.keras.models.load_model('results-transfer/ResNet50V2_line.h5'),
		 "ResNet50_infill" : tf.keras.models.load_model('results-transfer/ResNet50V2_infill.h5'),
         "Xception_line" : tf.keras.models.load_model('results-transfer/Xception_line.h5'),
		 "Xception_infill" : tf.keras.models.load_model('results-transfer/Xception_infill.h5')}
batch_size = 64
img_h = 224
img_w = 224
dataset_path_test1 = 'real_data/augmented_real_line/test'
dataset_path_test2 = 'real_data/augmented_real_infill/test'
#dataset_path_test = 'real_data/augmented_real_line'
line_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path_test1,
    label_mode="categorical",
    seed=1337,
    image_size=(img_h, img_w),
    batch_size=batch_size,
)
infill_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path_test2,
    label_mode="categorical",
    seed=1337,
    image_size=(img_h, img_w),
    batch_size=batch_size,
)


class_names=["good","over","under"]
results = []
for m in models:
    if "line" in m:
        test_ds = line_ds
    else:
        test_ds = infill_ds

    num_correct = 0

    for repeat in range(1):
        test_images = []
        test_labels = []
        #for test in range(100):
        for x,y in test_ds.as_numpy_iterator(): 
            if len(test_images) <= 1000:
                for i in range(batch_size):
                    image = x[i].astype("uint8")
                    test_images.append(image)
                    test_labels.append(y[i])

        for num, t_img in enumerate(test_images[1000*repeat:1000*(repeat+1)]):  

            predictions = models[m].predict(np.expand_dims(t_img, axis=0), verbose = 0)
            score = tf.nn.softmax(predictions[0])

            

            if np.argmax(score) == np.argmax(test_labels[num]):
                num_correct += 1

        
        print(f"{m} got {num_correct} out of 1000 correct")
        results.append(num_correct)

print(results)


