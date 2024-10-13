import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.cluster import KMeans 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from scipy.spatial.distance import cdist 
# Load Iris dataset 
iris = datasets.load_iris() 
X = iris.data 
y = iris.target 
# Standardize the dataset 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Split the dataset into 70% train and 30% test 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, 
random_state=42, stratify=y) 
# Function to compute accuracy of k-means clustering 
def kmeans_clustering(k, X_train, X_test, y_train, y_test): 
# Train the KMeans model with explicit n_init to suppress the warning 
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42) 
kmeans.fit(X_train) 
# Predict the labels for train and test data 
    y_train_pred = kmeans.predict(X_train) 
    y_test_pred = kmeans.predict(X_test) 
 
    # Adjust predicted clusters to match the true labels 
    def adjust_labels(y_pred, y_true): 
        label_map = {} 
        for i in np.unique(y_pred): 
            mask = (y_pred == i) 
            label_map[i] = np.bincount(y_true[mask]).argmax() 
        return np.array([label_map[label] for label in y_pred]) 
     
    y_train_pred_adj = adjust_labels(y_train_pred, y_train) 
    y_test_pred_adj = adjust_labels(y_test_pred, y_test) 
 
    # Calculate accuracy 
    train_accuracy = accuracy_score(y_train, y_train_pred_adj) 
    test_accuracy = accuracy_score(y_test, y_test_pred_adj) 
 
    return train_accuracy, test_accuracy 
 
# Test the effect of different k values (number of clusters) 
k_values = [2, 3, 4, 5] 
train_accuracies = [] 
test_accuracies = [] 
 
for k in k_values: 
    train_acc, test_acc = kmeans_clustering(k, X_train, X_test, y_train, y_test) 
    train_accuracies.append(train_acc) 
    test_accuracies.append(test_acc) 
    print(f"K={k}: Train Accuracy={train_acc:.4f}, Test Accuracy={test_acc:.4f}") 
 
# Plot the results 
plt.figure(figsize=(8, 5)) 
plt.plot(k_values, train_accuracies, label='Train Accuracy', marker='o') 
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='o') 
plt.xlabel('Number of Clusters (k)') 
plt.ylabel('Accuracy') 
plt.title('K-means Clustering Accuracy with Different k') 
plt.legend() 
plt.grid(True) 
plt.show() 
 
# Try other distance metrics: e.g., 'cityblock' (Manhattan), 'cosine' 
def kmeans_with_different_distance(X_train, X_test, y_train, y_test, 
metric='euclidean'): 
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42) 
    kmeans.fit(X_train) 
 
    y_test_pred = kmeans.predict(X_test) 
 
    # Adjust predicted clusters to match the true labels 
    def adjust_labels(y_pred, y_true): 
        label_map = {} 
        for i in np.unique(y_pred): 
            mask = (y_pred == i) 
            label_map[i] = np.bincount(y_true[mask]).argmax() 
        return np.array([label_map[label] for label in y_pred]) 
 
    y_test_pred_adj = adjust_labels(y_test_pred, y_test) 
# Calculate accuracy 
test_accuracy = accuracy_score(y_test, y_test_pred_adj) 
return test_accuracy 
# Manhattan and Cosine distances 
print(f"Test Accuracy (Manhattan): {kmeans_with_different_distance(X_train, X_test, 
y_train, y_test, metric='cityblock'):.4f}") 
print(f"Test Accuracy (Cosine): {kmeans_with_different_distance(X_train, X_test, 
y_train, y_test, metric='cosine'):.4f}") 
Output: 
K=2: Train Accuracy=0.6667, Test Accuracy=0.6667 
K=3: Train Accuracy=0.8667, Test Accuracy=0.7556 
K=4: Train Accuracy=0.8667, Test Accuracy=0.7556 
K=5: Train Accuracy=0.8571, Test Accuracy=0.7778 
Test Accuracy (Manhattan): 0.7556 
Test Accuracy (Cosine): 0.7556 
