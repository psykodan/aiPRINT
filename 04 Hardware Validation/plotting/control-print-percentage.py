import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Combined dataset for Line Over Extrusion, Line Under Extrusion, Infill Over Extrusion, and Infill Under Extrusion Tests
combined_data_with_infill_under = {
    "good": [65.38,60.61,53.85,63.64,69.7,74.29,50,71.43,62.5,77.14,90.55,94.4,90.85,92.36,89.86,89.52,88.71,90.85,92.68,88.03],
    "over": [26.92,30.3,30.77,27.27,21.21,0,0,0,0,0,7.87,4.8,7.19,6.25,6.76,3.23,0.81,1.31,1.63,1.41],
    "under": [7.69,9.09,15.38,9.09,9.09,25.71,50,28.57,37.5,22.86,1.57,0.8,1.96,1.39,3.38,7.26,10.48,7.84,5.69,10.56]
}

# Labels for the dataset
new_labels = ['Line Over Extrusion'] * 5 + ['Line Under Extrusion'] * 5 + ['Infill Over Extrusion'] * 5 + ['Infill Under Extrusion'] * 5

# Prepare the combined data in long format
combined_data_long_with_infill_under = pd.DataFrame({
    'Category': ['good'] * 20 + ['over'] * 20 + ['under'] * 20,
    'Values': combined_data_with_infill_under["good"] + combined_data_with_infill_under["over"] + combined_data_with_infill_under["under"],
    'Dataset': new_labels * 3
})

# Highlighted points for the control data
highlighted_points_with_infill_under = {
    "good": [48.48, 42.86, 69.28, 85.33],  # Control values for Line Over, Line Under, Infill Over, and Infill Under
    "over": [48.48, 0, 30.07, 0.67],
    "under": [3.03, 57.14, 0.65, 14]
}

# Prepare the highlighted points dataframe with the updated control data
highlighted_points_long_with_infill_under = pd.DataFrame({
    'Category': ['good'] * 4 +['over'] * 4+['under'] * 4,
    'Values': highlighted_points_with_infill_under["good"] + highlighted_points_with_infill_under["over"] + highlighted_points_with_infill_under["under"],
    'Label': ['Line Over Extrusion No Correction', 
              'Line Under Extrusion No Correction', 
              'Infill Over Extrusion No Correction', 
              'Infill Under Extrusion No Correction',]*3
})
print(highlighted_points_long_with_infill_under)

# Plot with jitter for better readability
fig, ax = plt.subplots(figsize=(6, 6))

# Plot each test data with jitter
sns.stripplot(x='Category', y='Values', data=combined_data_long_with_infill_under,
              ax=ax, size=10, edgecolor="black", linewidth=1, jitter=False, dodge=True, hue='Dataset',palette='rocket')
sns.stripplot(x='Category', y='Values', data=highlighted_points_long_with_infill_under,
              ax=ax, size=10, color="white", linewidth=2, jitter=False, dodge=True, hue='Label',marker="D",palette='crest')

'''
sns.stripplot(x='Category', y='Values', data=combined_data_long_with_infill_under[combined_data_long_with_infill_under['Dataset'] == 'Line Over Extrusion Test'],
              ax=ax, size=8, edgecolor="black", linewidth=1, jitter=False, dodge=True,color="darkorange", label="Line Over Extrusion")

sns.stripplot(x='Category', y='Values', data=combined_data_long_with_infill_under[combined_data_long_with_infill_under['Dataset'] == 'Line Under Extrusion Test'],
              ax=ax, size=8, edgecolor="black", linewidth=1, jitter=False, dodge=True,color="darkgreen", label="Line Under Extrusion")

sns.stripplot(x='Category', y='Values', data=combined_data_long_with_infill_under[combined_data_long_with_infill_under['Dataset'] == 'Infill Over Extrusion Test'],
              ax=ax, size=8, edgecolor="black", linewidth=1, jitter=False, dodge=True,color="purple", label="Infill Over Extrusion")

sns.stripplot(x='Category', y='Values', data=combined_data_long_with_infill_under[combined_data_long_with_infill_under['Dataset'] == 'Infill Under Extrusion Test'],
              ax=ax, size=8, edgecolor="black", linewidth=1, jitter=False, dodge=True,color="cyan", label="Infill Under Extrusion")

# Plot the highlighted control points with distinct markers and labels
for label, marker, color in zip(['Line Over Extrusion No Correction', 'Line Under Extrusion No Correction', 'Infill Over Extrusion No Correction', 'Infill Under Extrusion No Correction'],
                                ['D', 's', '^', 'p'], ['blue', 'green', 'red', 'cyan']):
    subset = highlighted_points_long_with_infill_under[highlighted_points_long_with_infill_under['Label'] == label]
    ax.scatter(subset['Category'], subset['Values'], color=color, marker=marker, zorder=5, label=label)
'''
# Set y-axis limits
ax.set_ylim(0, 100)

# Add custom labels for the x and y axis
ax.set_xlabel('Extrusion Classification')
ax.set_ylabel('Percentage Printed')

# Add a title
ax.set_title('Resulting Print Correction Using Control System')
'''
# Simplified legend (one entry per test and control point)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Line Over Extrusion', markerfacecolor='darkorange', markersize=10),
    plt.Line2D([0], [0], marker='D', color='w', label='Line Over Extrusion No Correction', markerfacecolor='blue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Line Under Extrusion', markerfacecolor='darkgreen', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', label='Line Under Extrusion No Correction', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Infill Over Extrusion', markerfacecolor='purple', markersize=10),
    plt.Line2D([0], [0], marker='^', color='w', label='Infill Over Extrusion No Correction', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Infill Under Extrusion', markerfacecolor='cyan', markersize=10),
    plt.Line2D([0], [0], marker='p', color='w', label='Infill Under Extrusion No Correction', markerfacecolor='cyan', markersize=10)
]
ax.legend(handles=legend_elements)
'''

# Add vertical gridlines
y_ticks = [10,20,30,40,50,60,70,80,90,100]
plt.yticks(y_ticks)
for x in y_ticks:
    plt.axhline(y=x, color='gray', linestyle='--', linewidth=0.5)


plt.legend(loc=0)
# Show the plot
plt.tight_layout()
#plt.show()
plt.savefig("cnn-validation-print-percent.png",dpi=300)
