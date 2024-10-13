# Import necessary libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
 
# Define the dataset 
data = { 
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 
'Sunny', 'Rainy',  
                'Sunny', 'Overcast', 'Overcast', 'Rainy'], 
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 
'Mild', 'Hot', 'Mild'], 
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 
'Normal', 'Normal', 'Normal',  
                 'High', 'Normal', 'High'], 
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, 
False, True], 
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'] 
} 
 
# Convert the dataset to a DataFrame 
df = pd.DataFrame(data) 
 
# Encode categorical variables 
df_encoded = pd.get_dummies(df.drop('Play', axis=1)) 
y = df['Play'].map({'No': 0, 'Yes': 1})  # Encode target variable 
 
# Split the dataset (Randomly select 10 training samples and 4 test samples) 
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, train_size=10, 
random_state=42, shuffle=True) 
# Initialize the Decision Tree Classifier 
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42) 
# Train the model 
clf.fit(X_train, y_train) 
# Make predictions on the test set 
y_pred = clf.predict(X_test) 
# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
conf_matrix = confusion_matrix(y_test, y_pred) 
report = classification_report(y_test, y_pred, target_names=['No', 'Yes']) 
# Print the evaluation results 
print(f"Accuracy: {accuracy:.2f}") 
print("\nConfusion Matrix:\n", conf_matrix) 
print("\nClassification Report:\n", report) 
Output: 
Accuracy: 0.75 
Confusion Matrix: 
[[1 0] 
[1 2]] 
Classification Report: 
                precision     recall    f1-score    support 
 
          No       0.50       1.00        0.67          1 
         Yes        1.00       0.67       0.80          3 
 
    accuracy                              0.75          4 
   macro avg       0.75       0.83       0.73          4 
weighted avg       0.88       0.75       0.77          4 
 
 
 
