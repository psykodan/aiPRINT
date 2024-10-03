import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random


model = tf.keras.models.load_model('results-real/vgg19_line.h5')
batch_size = 64
img_h = 224
img_w = 224
dataset_path_test = 'real_data/augmented_real_infill/test/'
dataset_path_test = 'real_data/augmented_real_line/test/'
test_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path_test,
    label_mode="categorical",
    seed=1678,
    image_size=(img_h, img_w),
    batch_size=batch_size,
)

class_names=["good","over","under"]

num_correct = 0
fig = plt.figure(figsize=(8,16))

for repeat in range(1):
    test_images = []
    test_labels = []
    #for test in range(100):
    for x,y in test_ds.as_numpy_iterator(): 
        if len(test_images) <= 8:
            for i in range(batch_size):
                image = x[i].astype("uint8")
                test_images.append(image)
                test_labels.append(y[i])

    for num, t_img in enumerate(test_images[8*repeat:8*(repeat+1)]):  
        
        y = fig.add_subplot(4, 2, num + 1)


        predictions = model.predict(np.expand_dims(t_img, axis=0), verbose = 0)
        score = tf.nn.softmax(predictions[0])

        
        str_label = class_names[np.argmax(score)]
        if np.argmax(score) == np.argmax(test_labels[num]):
            num_correct += 1
            plt.title(str_label,fontsize = 40)
        else:
            plt.title(str_label,color="red",fontsize = 40)
        
        y.imshow(t_img)
        
        

        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    
    print(f"{num_correct} out of 16 correct")
plt.tight_layout()
plt.savefig("line.png",dpi=300)