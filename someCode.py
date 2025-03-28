import numpy as np
from sklearn.metrics import confusion_matrix

# Example data
# Actual labels
y_true = [2, 0, 2, 2, 0, 1]

# Predicted labels
y_pred = [0, 0, 2, 2, 0, 2]

# Generating the confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
