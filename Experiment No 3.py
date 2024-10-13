import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
# Load the Iris dataset 
iris = datasets.load_iris() 
X = iris.data  # Features (4-dimensional) 
y = iris.target  # Target (class labels) 
target_names = iris.target_names  # Class names 
# Print dataset description 
print(iris.DESCR) 
# Study the data 
print("Feature names:", iris.feature_names) 
print("First 5 rows of the data:\n", X[:5]) 
print("Class labels:", np.unique(y)) 
# Standardizing the features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Applying PCA to reduce dimensions to 2 
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled) 
# Check the explained variance ratio 
print("Explained variance ratio:", pca.explained_variance_ratio_) 
# Create a scatter plot of the two principal components 
plt.figure(figsize=(8, 6)) 
colors = ['navy', 'turquoise', 'darkorange'] 
for color, i, target_name in zip(colors, [0, 1, 2], target_names): 
plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, lw=2, 
label=target_name) 
plt.legend(loc='best', shadow=False, scatterpoints=1) 
plt.title('PCA of Iris Dataset') 
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Component 2') 
plt.grid(True) 
plt.show() 
