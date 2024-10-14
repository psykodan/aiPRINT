import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
replace2linear = ReplaceToLinear()


from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([0,1,2])

models ={"Xception_infill_synth" : tf.keras.models.load_model('05 Data/models/results-synth/Xception_infill.h5'),
         "Xception_line_synth" : tf.keras.models.load_model('05 Data/models/results-synth/Xception_line.h5'),
         "Xception_infill_real" : tf.keras.models.load_model('05 Data/models/results-real/Xception_infill.h5'),
         "Xception_line_real" : tf.keras.models.load_model('05 Data/models/results-real/Xception_line.h5'),
         "Xception_infill_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/Xception_infill.h5'),
         "Xception_line_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/Xception_line.h5'),
         "ResNet50V2_infill_synth" : tf.keras.models.load_model('05 Data/models/results-synth/ResNet50V2_infill.h5'),
         "ResNet50V2_line_synth" : tf.keras.models.load_model('05 Data/models/results-synth/ResNet50V2_line.h5'),
         "ResNet50V2_infill_real" : tf.keras.models.load_model('05 Data/models/results-real/ResNet50V2_infill.h5'),
         "ResNet50V2_line_real" : tf.keras.models.load_model('05 Data/models/results-real/ResNet50V2_line.h5'),
         "ResNet50V2_infill_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/ResNet50V2_infill.h5'),
         "ResNet50V2_line_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/ResNet50V2_line.h5'),
         "MobileNetV3Small_infill_synth" : tf.keras.models.load_model('05 Data/models/results-synth/MobileNetV3Small_infill.h5'),
         "MobileNetV3Small_line_synth" : tf.keras.models.load_model('05 Data/models/results-synth/MobileNetV3Small_line.h5'),
         "MobileNetV3Small_infill_real" : tf.keras.models.load_model('05 Data/models/results-real/MobileNetV3Small_infill.h5'),
         "MobileNetV3Small_line_real" : tf.keras.models.load_model('05 Data/models/results-real/MobileNetV3Small_line.h5'),
         "MobileNetV3Small_infill_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/MobileNetV3Small_infill.h5'),
         "MobileNetV3Small_line_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/MobileNetV3Small_line.h5'),
         "vgg19_infill_synth" : tf.keras.models.load_model('05 Data/models/results-synth/vgg19_infill.h5'),
         "vgg19_line_synth" : tf.keras.models.load_model('05 Data/models/results-synth/vgg19_line.h5'),
         "vgg19_infill_real" : tf.keras.models.load_model('05 Data/models/results-real/vgg19_infill.h5'),
         "vgg19_line_real" : tf.keras.models.load_model('05 Data/models/results-real/vgg19_line.h5'),
         "vgg19_infill_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/vgg19_infill.h5'),
         "vgg19_line_transfer" : tf.keras.models.load_model('05 Data/models/results-transfer/vgg19_line.h5'),
         
         }
# Image titles
image_titles = ['good', 'over', 'under']

# Load images and Convert them to a Numpy array
img1 = load_img('05 Data/datasets/synth_data/augmented_synth_line/test/good/100.jpg', target_size=(224, 224))
img2 = load_img('05 Data/datasets/synth_data/augmented_synth_infill/test/good/100.jpg', target_size=(224, 224))
line_img_synth = np.asarray([np.array(img1),np.array(img1),np.array(img1)])
infill_img_synth = np.asarray([np.array(img2),np.array(img2),np.array(img2)])

# Load images and Convert them to a Numpy array
img1 = load_img('05 Data/datasets/real_data/augmented_real_line/test/good/100.jpg', target_size=(224, 224))
img2 = load_img('05 Data/datasets/real_data/augmented_real_infill/test/good/100.jpg', target_size=(224, 224))
line_img_real = np.asarray([np.array(img1),np.array(img1),np.array(img1)])
infill_img_real = np.asarray([np.array(img2),np.array(img2),np.array(img2)])


X={"Xception_infill_synth" : tf.keras.applications.xception.preprocess_input(infill_img_synth).astype(np.float32),
         "Xception_line_synth" : tf.keras.applications.xception.preprocess_input(line_img_synth).astype(np.float32),
         "Xception_infill_real" : tf.keras.applications.xception.preprocess_input(infill_img_real).astype(np.float32),
         "Xception_line_real" : tf.keras.applications.xception.preprocess_input(line_img_real).astype(np.float32),
         "Xception_infill_transfer" : tf.keras.applications.xception.preprocess_input(infill_img_real).astype(np.float32),
         "Xception_line_transfer" : tf.keras.applications.xception.preprocess_input(line_img_real).astype(np.float32),
         "ResNet50V2_infill_synth" : tf.keras.applications.resnet_v2.preprocess_input(infill_img_synth).astype(np.float32),
         "ResNet50V2_line_synth" : tf.keras.applications.resnet_v2.preprocess_input(line_img_synth).astype(np.float32),
         "ResNet50V2_infill_real" : tf.keras.applications.resnet_v2.preprocess_input(infill_img_real).astype(np.float32),
         "ResNet50V2_line_real" : tf.keras.applications.resnet_v2.preprocess_input(line_img_real).astype(np.float32),
         "ResNet50V2_infill_transfer" : tf.keras.applications.resnet_v2.preprocess_input(infill_img_real).astype(np.float32),
         "ResNet50V2_line_transfer" :tf.keras.applications.resnet_v2.preprocess_input(line_img_real).astype(np.float32),
         "MobileNetV3Small_infill_synth" : tf.keras.applications.mobilenet_v3.preprocess_input(infill_img_synth).astype(np.float32),
         "MobileNetV3Small_line_synth" : tf.keras.applications.mobilenet_v3.preprocess_input(line_img_synth).astype(np.float32),
         "MobileNetV3Small_infill_real" :tf.keras.applications.mobilenet_v3.preprocess_input(infill_img_real).astype(np.float32),
         "MobileNetV3Small_line_real" : tf.keras.applications.mobilenet_v3.preprocess_input(line_img_real).astype(np.float32),
         "MobileNetV3Small_infill_transfer" : tf.keras.applications.mobilenet_v3.preprocess_input(infill_img_real).astype(np.float32),
         "MobileNetV3Small_line_transfer" : tf.keras.applications.mobilenet_v3.preprocess_input(infill_img_real).astype(np.float32),
         "vgg19_infill_synth" : tf.keras.applications.vgg19.preprocess_input(infill_img_synth),
         "vgg19_line_synth" : tf.keras.applications.vgg19.preprocess_input(line_img_synth),
         "vgg19_infill_real" : tf.keras.applications.vgg19.preprocess_input(infill_img_real),
         "vgg19_line_real" : tf.keras.applications.vgg19.preprocess_input(line_img_real),
         "vgg19_infill_transfer" : tf.keras.applications.vgg19.preprocess_input(infill_img_real),
         "vgg19_line_transfer" : tf.keras.applications.vgg19.preprocess_input(line_img_real),
         
         }



saliency_maps = []
for m in models:
    # Create Saliency object.
    print(m)
    saliency = Saliency(models[m],
                        model_modifier=replace2linear,
                        clone=True)

    # Generate saliency map
    saliency_maps.append(saliency(score, X[m]))
    

# Render
f, ax = plt.subplots(nrows=int(len(models)/6), ncols=6, figsize=(6,(len(models)/6)))
y=0
labels = ['Xception','ResNet50V2','MobileNetV3','VGG19']
ax[0][0].set_title("Infill",rotation=45)
ax[0][1].set_title("Line",rotation=45)
ax[0][2].set_title("Infill",rotation=45)
ax[0][3].set_title("Line",rotation=45)
ax[0][4].set_title("Infill",rotation=45)
ax[0][5].set_title("Line",rotation=45)

r=2
for idx,m in enumerate(models):
    
    print(m)
    if "infill" in m:
        if "synth" in m:
            images = infill_img_synth
            ax[y][0].imshow(saliency_maps[idx][0], cmap='jet')
            ax[y][0].set_xticks([])
            ax[y][0].set_yticks([])
            #ax[y][0].set_ylabel(labels[y], rotation=45)
            #ax[y][0].yaxis.set_label_coords(-0.7, 0.5)
        else:
            images = infill_img_real
            ax[y][r].imshow(saliency_maps[idx][0], cmap='jet')
            ax[y][r].set_xticks([])
            ax[y][r].set_yticks([])
            r+=1
            #ax[y][r].set_ylabel(labels[y], rotation=45)
            #ax[y][r].yaxis.set_label_coords(-0.7, 0.5)
        
    else:
        if "synth" in m:
            images = line_img_synth
            ax[y][1].imshow(saliency_maps[idx][0], cmap='jet')
            ax[y][1].set_xticks([])
            ax[y][1].set_yticks([])
        else:
            images = line_img_real
            ax[y][r].imshow(saliency_maps[idx][0], cmap='jet')
            ax[y][r].set_xticks([])
            ax[y][r].set_yticks([])
            r+=1
    
    if r>=6:
        y+=1
        r=2

#plt.tight_layout()
#plt.show()
plt.savefig("attentions.png",dpi=300)
