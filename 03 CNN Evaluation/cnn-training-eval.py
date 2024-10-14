import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

metrics_line = []
metrics_infill = []
model_line = [tf.keras.models.load_model('05 Data/models/results-synth/vgg19_line.h5'),
              tf.keras.models.load_model('05 Data/models/results-real/vgg19_line.h5'),
              tf.keras.models.load_model('05 Data/models/results-transfer/vgg19_line.h5')]
model_infill = [tf.keras.models.load_model('05 Data/models/results-synth/vgg19_infill.h5'),
                tf.keras.models.load_model('05 Data/models/results-real/vgg19_infill.h5'),
                tf.keras.models.load_model('05 Data/models/results-transfer/vgg19_infill.h5')]
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
    test_ds_line = test_ds_line.take(100)
    test_ds_infill = test_ds_infill.take(100)



    res = model_line[run].evaluate(test_ds_line, verbose=2)
    metrics_line.append(dict(zip(model_line[run].metrics_names, res)))
    metrics_line[-1]["f1"] = (2 * metrics_line[-1]["precision"] * metrics_line[-1]["recall"]) / (metrics_line[-1]["precision"] + metrics_line[-1]["recall"])
    print(metrics_line)

    res = model_infill[run].evaluate(test_ds_infill, verbose=2)
    metrics_infill.append(dict(zip(model_infill[run].metrics_names, res)))
    if run == 2:
        metrics_infill[-1]["f1"] = 0
    else:
        metrics_infill[-1]["f1"] = (2 * metrics_infill[-1]["precision"] * metrics_infill[-1]["recall"]) / (metrics_infill[-1]["precision"] + metrics_infill[-1]["recall"])
    print(metrics_infill)

value = []
for i in range(3):
    value.append([metrics_line[i]['accuracy'],metrics_infill[i]['accuracy'],metrics_line[i]['f1'],metrics_infill[i]['f1'],
              metrics_line[i]['precision'],metrics_infill[i]['precision'],metrics_line[i]['recall'],metrics_infill[i]['recall']])
    
import matplotlib.pyplot as plt
import numpy as np

# Sample data for three groups (infill and line) for the 4 metrics
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
infill_values_group_A = [value[0][1]*100,value[0][3]*100,value[0][5]*100,value[0][7]*100]  # Values for infill for Group A (Synthetic)
line_values_group_A = [value[0][0]*100,value[0][2]*100,value[0][4]*100,value[0][6]*100]  # Values for line for Group A (Synthetic)

infill_values_group_B = [value[1][1]*100,value[1][3]*100,value[1][5]*100,value[1][7]*100]  # Values for infill for Group B (Real)
line_values_group_B = [value[1][0]*100,value[1][2]*100,value[1][4]*100,value[1][6]*100]  # Values for line for Group B (Real)

infill_values_group_C = [value[2][1]*100,value[2][3]*100,value[2][5]*100,value[2][7]*100]  # Values for infill for Group C (Transfer)
line_values_group_C = [value[2][0]*100,value[2][2]*100,value[2][4]*100,value[2][6]*100]  # Values for line for Group C (Transfer)

# Create a range for the y-axis (the number of metrics)
y_pos = np.arange(len(metrics))

# Set the width of the bars
bar_width = 0.2

# Create the diverging bar plot with three groups
plt.figure(figsize=(5, 5))

# Plot Group A: Infill (filled) and Line (border only)
bar_infill_A = plt.barh(y_pos - bar_width, -np.array(infill_values_group_A), bar_width, color='cornflowerblue', align='center')
bar_line_A = plt.barh(y_pos - bar_width, line_values_group_A, bar_width, facecolor='none', edgecolor='royalblue', align='center', linewidth=2)

# Plot Group B: Infill (filled) and Line (border only)
bar_infill_B = plt.barh(y_pos, -np.array(infill_values_group_B), bar_width, color='lightgreen', align='center')
bar_line_B = plt.barh(y_pos, line_values_group_B, bar_width, facecolor='none', edgecolor='green', align='center', linewidth=2)

# Plot Group C: Infill (filled) and Line (border only)
bar_infill_C = plt.barh(y_pos + bar_width, -np.array(infill_values_group_C), bar_width, color='lightcoral', align='center')
bar_line_C = plt.barh(y_pos + bar_width, line_values_group_C, bar_width, facecolor='none', edgecolor='darkred', align='center', linewidth=2)

# Add the metric labels to the y-axis
plt.yticks(y_pos, metrics)

# Add labels for the x-axis
plt.xlabel('Scores (%)')

# Add a title
plt.title('VGG19')

# Add label to indicate "Infill" (left) and "Line" (right)
plt.text(-55, len(metrics), 'Infill', fontsize=12, ha='center', va='center')
plt.text(55, len(metrics), 'Line', fontsize=12, ha='center', va='center')

# Custom x-axis ticks with rounded values at intervals of 0.05 and rotated 90 degrees
x_ticks = np.arange(-100, 105, 10)
rounded_ticks = [round(abs(x), 0) for x in x_ticks]
plt.xticks(x_ticks, labels=rounded_ticks, rotation=45)

# Add value labels to each bar
for bars, infill_values, line_values in zip(
    [bar_infill_A, bar_infill_B, bar_infill_C], 
    [infill_values_group_A, infill_values_group_B, infill_values_group_C], 
    [line_values_group_A, line_values_group_B, line_values_group_C]
):
    for bar, infill_value, line_value in zip(bars, infill_values, line_values):
        # Label for infill (left side of the plot)
        if infill_value < 10:
            plt.text(bar.get_width() - 10.22, (bar.get_y() + bar.get_height() / 2)-0.02, 
                 f'N/A', va='center', ha='right', color='black', fontsize=10)
        else: 
            plt.text(bar.get_width() + 30.22, (bar.get_y() + bar.get_height() / 2)-0.02, 
                 f'{abs(infill_value):.2f}', va='center', ha='right', color='black', fontsize=10)
        # Label for line (right side of the plot)
        plt.text(line_value - 30.22, (bar.get_y() + bar.get_height() / 2)-0.02, 
                 f'{line_value:.2f}', va='center', ha='left', color='black', fontsize=10)


# Add vertical gridlines
for x in x_ticks:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

# Custom legend
import matplotlib.patches as mpatches

# Create custom patches for the legend, representing the three groups by color
synthetic_patch = mpatches.Patch(color='cornflowerblue', label='Synthetic')
real_patch = mpatches.Patch(color='lightgreen', label='Real')
transfer_patch = mpatches.Patch(color='lightcoral', label='Transfer')

# Adjust the legend position to avoid overlapping the bars
plt.legend(handles=[synthetic_patch, real_patch, transfer_patch], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# Show the plot
plt.tight_layout()
#plt.show()
plt.savefig("cnn-evaluation-vgg19.png",dpi=300)