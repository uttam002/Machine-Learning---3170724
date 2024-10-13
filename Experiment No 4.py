import numpy as np 
import matplotlib.pyplot as plt 
# Given Vectors 
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
B = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1, 0]) 
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
Y = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1, 0]) 
# Function to calculate Cosine Similarity 
def cosine_similarity(vec1, vec2): 
dot_product = np.dot(vec1, vec2) 
magnitude_vec1 = np.linalg.norm(vec1) 
magnitude_vec2 = np.linalg.norm(vec2) 
return dot_product / (magnitude_vec1 * magnitude_vec2) 
# Function to calculate Euclidean Distance 
def euclidean_distance(vec1, vec2): 
return np.linalg.norm(vec1 - vec2) 
# Calculating Cosine Similarity and Euclidean Distance for A, B and X, Y 
cosine_sim_AB = cosine_similarity(A, B) 
euclidean_dist_AB = euclidean_distance(A, B) 
cosine_sim_XY = cosine_similarity(X, Y) 
euclidean_dist_XY = euclidean_distance(X, Y) 
print(f"Cosine Similarity between A and B: {cosine_sim_AB}") 
print(f"Euclidean Distance between A and B: {euclidean_dist_AB}") 
print(f"Cosine Similarity between X and Y: {cosine_sim_XY}") 
print(f"Euclidean Distance between X and Y: {euclidean_dist_XY}") 
# Geometric explanation of Cosine Similarity and Euclidean Distance 
# This is a simple 2D representation for illustration purposes 
# Create a simple plot to represent vectors A and B 
plt.figure(figsize=(10, 5)) 
# Plot vectors A and B in a 2D space (for illustrative purposes) 
plt.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector A') 
plt.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector B') 
# Annotate the cosine similarity and euclidean distance 
plt.text(A[0] / 2, A[1] / 2, 'A', fontsize=12, color='red') 
plt.text(B[0] / 2, B[1] / 2, 'B', fontsize=12, color='blue') 
# Add the Euclidean distance as a dashed line 
plt.plot([A[0], B[0]], [A[1], B[1]], 'k--', label='Euclidean Distance') 
# Setting up the plot 
plt.xlim(-1, max(A[0], B[0]) + 1) 
plt.ylim(-1, max(A[1], B[1]) + 1) 
plt.grid() 
plt.axhline(0, color='black', linewidth=0.5) 
plt.axvline(0, color='black', linewidth=0.5) 
plt.title("Geometric Representation of Cosine Similarity and Euclidean Distance for A and B") 
plt.legend() 
plt.show() 
# Repeat the same for X and Y 
plt.figure(figsize=(10, 5)) 
# Plot vectors X and Y in a 2D space (for illustrative purposes) 
plt.quiver(0, 0, X[0], X[1], angles='xy', scale_units='xy', scale=1, color='g', label='Vector X') 
plt.quiver(0, 0, Y[0], Y[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Vector Y') 
# Annotate the cosine similarity and euclidean distance 
plt.text(X[0] / 2, X[1] / 2, 'X', fontsize=12, color='green') 
plt.text(Y[0] / 2, Y[1] / 2, 'Y', fontsize=12, color='orange') 
# Add the Euclidean distance as a dashed line 
plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k--', label='Euclidean Distance') 
# Setting up the plot 
plt.xlim(-1, max(X[0], Y[0]) + 1) 
plt.ylim(-1, max(X[1], Y[1]) + 1) 
plt.grid() 
plt.axhline(0, color='black', linewidth=0.5) 
plt.axvline(0, color='black', linewidth=0.5) 
plt.title("Geometric Representation of Cosine Similarity and Euclidean Distance for X and Y") 
plt.legend() 
plt.show() 
Output:  
Cosine Similarity between A and B: 0.6621003580727145 
Euclidean Distance between A and B: 14.966629547095765 
Cosine Similarity between X and Y: 0.6621003580727145 
Euclidean Distance between X and Y: 14.966629547095765 
