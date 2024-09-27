import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Set values of k
k_values = [1, 3, 5, 7]
accuracy_results = []

# Perform 10-fold cross-validation for each k
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X, y, cv=kf, scoring='accuracy')
    accuracy_results.append(scores.mean())

# Display the results
print("Value of k\tAccuracy of model")
for k, accuracy in zip(k_values, accuracy_results):
    print(f"{k}\t\t{accuracy:.4f}")

# Plotting k vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_results, marker='o')
plt.title('k-NN Classifier Accuracy')
plt.xlabel('Value of k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.ylim(0, 1)  # Set y-axis from 0 to 1
plt.grid()
plt.show()

# output
# Value of k    Accuracy of model
#   1             0.9667

#   3             0.9667
#   5             0.9667
#   7             0.9667
