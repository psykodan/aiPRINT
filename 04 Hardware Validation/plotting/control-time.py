import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for Over Extrusion and Under Extrusion
detection_over_extrusion = [1.734693878, 1.176470588, 1.041666667, 1.176470588, 0.666666667]
correction_over_extrusion = [10.71428571, 12.60504202, 12.5, 10.08403361, 8.75]

detection_under_extrusion = [2.08, 3.404255319, 2.936507937, 2.564102564, 4.881889764]
correction_under_extrusion = [7.2, 15.95744681, 9.523809524, 12.82051282, 7.086614173]

# Organize the data into a dataframe for easy plotting with boxplots
data = {
    'Set': ['Over Extrusion'] * 5 + ['Under Extrusion'] * 5 + ['Over Extrusion'] * 5 + ['Under Extrusion'] * 5,
    'Type': ['Detection'] * 10 + ['Correction'] * 10,
    'Time (s)': detection_over_extrusion + detection_under_extrusion + correction_over_extrusion + correction_under_extrusion
}

df = pd.DataFrame(data)

# Create the box plots
plt.figure(figsize=(3, 6))
sns.boxplot(x='Type', y='Time (s)', hue='Set', data=df, palette="flare")
y_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
plt.yticks(y_ticks)
for x in y_ticks:
    plt.axhline(y=x, color='gray', linestyle='--', linewidth=0.5)
# Add title and labels
plt.title('Time Until Error Detection \nand Correction')
plt.ylabel('Time (s)')
plt.xlabel('Measurement Type')
plt.legend(title=None)
# Show the plot
plt.tight_layout()
#plt.show()
plt.savefig("cnn-validation-error-time.png",dpi=300)
