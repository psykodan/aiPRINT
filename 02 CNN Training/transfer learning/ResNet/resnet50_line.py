import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import layers, Model
import pathlib
import json

#strategy = tf.distribute.MirroredStrategy()
#print('DEVICES AVAILABLE: {}'.format(strategy.num_replicas_in_sync))



data_dir = pathlib.Path("data/real_data/augmented_real_line/train")

batch_size = 64
img_h = 224
img_w = 224
epochs = 1000
history = None


#split data into train test (not validation) 0.2
train_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	seed = 123,
	image_size=(img_h, img_w),
	batch_size=batch_size,
	label_mode='categorical',
    subset="training",
    validation_split=0.2)

val_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	seed = 123,
	image_size=(img_h, img_w),
	batch_size=batch_size,
	label_mode='categorical',
    subset="validation",
    validation_split=0.2)

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)

class_num_training_samples = {}
for f in train_ds.file_paths:
    class_name = f.split('/')[-2]
    if class_name in class_num_training_samples:
        class_num_training_samples[class_name] += 1
    else:
        class_num_training_samples[class_name] = 1
max_class_samples = max(class_num_training_samples.values())
class_weights = {}
for i in range(0, len(train_ds.class_names)):
    class_weights[i] = max_class_samples/class_num_training_samples[train_ds.class_names[i]]

print(class_weights)
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10,  restore_best_weights=True)


name = "results/ResNet50V2_line"
with tf.device('/device:GPU:1'):
    '''
    inp = layers.Input((img_h,img_h,3))
    x = ResNet50V2(input_tensor=inp, weights=None, include_top=False )
    x.trainable = True ## Not trainable weights
    x = layers.Flatten()(x.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(3, activation='softmax')(x)
    model = Model(inp, x)

    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=["accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.FalseNegatives()
                ])
    '''
    model = tf.keras.models.load_model('results-synth/ResNet50V2_line.h5')
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=64, callbacks=[es],verbose = 2)
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(f'{name}.json', 'w'))
    model.save(f'{name}.h5')
    model.save(f'{name}.keras')
