import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler 
# Load Iris dataset 
iris = datasets.load_iris() 
X = iris.data 
y = iris.target 
class_labels = np.unique(y) 
# Standardize the data 
sc = StandardScaler() 
X_std = sc.fit_transform(X) 
# Compute the mean of all features 
mean_overall = np.mean(X_std, axis=0) 
# Class scatter matrix (Within-class scatter) 
S_W = np.zeros((X.shape[1], X.shape[1])) 
for label in class_labels: 
X_class = X_std[y == label] 
mean_class = np.mean(X_class, axis=0) 
S_W += (X_class - mean_class).T.dot(X_class - mean_class) 
# Between-class scatter matrix 
S_B = np.zeros((X.shape[1], X.shape[1])) 
for label in class_labels: 
X_class = X_std[y == label] 
mean_class = np.mean(X_class, axis=0) 
n_class = X_class.shape[0]  # number of samples in class 
mean_diff = (mean_class - mean_overall).reshape(-1, 1) 
S_B += n_class * (mean_diff).dot(mean_diff.T) 
# Total scatter matrix 
S_T = S_W + S_B 
# Print results 
print("Within-class scatter matrix (S_W):\n", S_W) 
print("\nBetween-class scatter matrix (S_B):\n", S_B) 
print("\nTotal scatter matrix (S_T):\n", S_T) 
Output: 
Within-class scatter matrix (S_W): 
[[57.19414039 38.01741201 16.95864185  9.00354437] 
[38.01741201 89.88257294 10.62508571 14.57007861] 
[16.95864185 10.62508571  8.79424214  4.69233162] 
[ 9.00354437 14.57007861  4.69233162 10.66756048]] 
Between-class scatter matrix (S_B): 
[[ 92.80585961 -55.65287963 113.80442453 113.68762457] 
[-55.65287963  60.11742706 -74.89110136 -69.48896849] 
[113.80442453 -74.89110136 141.20575786 139.73748309] 
[113.68762457 -69.48896849 139.73748309 139.33243952]] 
Total scatter matrix (S_T): 
[[150.         -17.63546762 130.76306638 122.69116894] 
[-17.63546762 150.         -64.26601565 -54.91888988] 
[130.76306638 -64.26601565 150.         
144.42981471] 
[122.69116894 -54.91888988 144.42981471 150.        
]] 
 
 
 
