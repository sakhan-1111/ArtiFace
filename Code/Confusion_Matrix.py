# Importing all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create and save confusion matrix
cm = [[3710, 3735], [2165, 5315]]
cm_df = pd.DataFrame(cm)
target_names = ['Computer-generated', 'Real']

plt.figure(figsize = (4, 4))
sns.heatmap(cm_df,
            cmap='gnuplot',
            fmt='d',
            annot = True,
            cbar=True,
            xticklabels = target_names,
            yticklabels = target_names)
plt.ylabel('True label', fontsize = 12)
plt.xlabel('Predicted label', fontsize = 12)
plt.title('Test confusion matrix', fontsize = 14)
# plt.savefig('Output_Confusion_Matrix.png')
plt.show()
