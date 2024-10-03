import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from skimage.transform import resize

#Line model
models ={"vgg19_line" : tf.keras.models.load_model('results-real/vgg19_line.h5'),
		 
         "MobileNetV3_line" : tf.keras.models.load_model('results-real/MobileNetV3Small_line.h5'),
		 
         "ResNet50_line" : tf.keras.models.load_model('results-real/ResNet50V2_line.h5'),
		
         "Xception_line" : tf.keras.models.load_model('results-real/Xception_line.h5')}
		 

img1 = load_img('real_data/augmented_real_line/test/good/400.jpg', target_size=(224, 224))

fig, ax = plt.subplots(8, 4, sharex=True, sharey=True, figsize=(4,8))
col = 0
for m in models:
    ax[0][col].set_title(f"{m}",rotation=45)
    # Extract output from each layer
    extractor = tf.keras.Model(inputs=models[m].inputs,
                            outputs=[layer.output for layer in models[m].layers])
    features = extractor(np.expand_dims(img1, 0))
    
    if len(features[1].numpy()[0][0][0]) > 4:
         # Show the 32 feature maps from the first layer
        l1_features = features[1].numpy()[0]
    else:
        l1_features = features[2].numpy()[0]
   
    ax[0][0].set_ylabel("first layer", rotation=45)
    ax[0][0].yaxis.set_label_coords(-0.7, 0.5)
    for i in range(0, 4):
        row = i//1
        ax[row][col].imshow(resize(l1_features[..., i],(224,224)), interpolation='nearest')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])

    # Show the 32 feature maps from the first layer
    l1_features = features[len(features)-5].numpy()[0]

    ax[4][0].set_ylabel("last layer", rotation=45)
    ax[4][0].yaxis.set_label_coords(-0.7, 0.5)
    for i in range(4, 8):
        row = i//1
        ax[row][col].imshow(resize(l1_features[..., i],(224,224)), interpolation='nearest')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
    
    col +=1
plt.tight_layout()
plt.savefig("features.png",dpi=300)
#plt.show()