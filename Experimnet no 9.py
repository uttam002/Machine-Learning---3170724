import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam, SGD, RMSprop 
from tensorflow.keras.utils import to_categorical 
# Load the Iris dataset 
iris = load_iris() 
X, y = iris.data, iris.target 
# Split the data into training (70%) and testing (30%) sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42) 
# Standardize the features 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
# Convert labels to one-hot encoded format 
y_train_encoded = to_categorical(y_train) 
y_test_encoded = to_categorical(y_test) 
def create_model(architecture, activation='relu', output_activation='softmax'): 
model = Sequential() 
for units in architecture: 
        model.add(Dense(units, activation=activation)) 
    model.add(Dense(3, activation=output_activation)) 
    return model 
 
def train_and_evaluate(model, optimizer_name, X_train, y_train, X_test, y_test, 
epochs=100, batch_size=32): 
    if optimizer_name == 'Adam': 
        optimizer = Adam(learning_rate=0.001) 
    elif optimizer_name == 'SGD': 
        optimizer = SGD(learning_rate=0.01) 
    elif optimizer_name == 'RMSprop': 
        optimizer = RMSprop(learning_rate=0.001) 
    else: 
        raise ValueError(f"Unknown optimizer: {optimizer_name}") 
     
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
metrics=['accuracy']) 
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
validation_split=0.2, verbose=0) 
    _, accuracy = model.evaluate(X_test, y_test, verbose=0) 
    return accuracy, history 
 
# Define different architectures and optimizers to try 
architectures = [ 
    [8, 8], 
    [16, 8], 
    [32, 16, 8], 
] 
 
optimizers = ['Adam', 'SGD', 'RMSprop'] 
 
results = [] 
 
for arch in architectures: 
    for optimizer_name in optimizers: 
        model = create_model(arch) 
        accuracy, history = train_and_evaluate(model, optimizer_name, X_train_scaled, 
y_train_encoded, X_test_scaled, y_test_encoded) 
        results.append({ 
            'architecture': arch, 
            'optimizer': optimizer_name, 
            'accuracy': accuracy, 
            'val_accuracy': max(history.history['val_accuracy']) 
        }) 
 
# Print results 
for result in results: 
    print(f"Architecture: {result['architecture']}, Optimizer: {result['optimizer']}") 
    print(f"Test Accuracy: {result['accuracy']:.4f}, Best Validation Accuracy: 
{result['val_accuracy']:.4f}") 
    print() 
 
# Find the best performing model 
best_model = max(results, key=lambda x: x['accuracy']) 
print("Best Model:") 
print(f"Architecture: {best_model['architecture']}, Optimizer: 
{best_model['optimizer']}") 
print(f"Test Accuracy: {best_model['accuracy']:.4f}") 
 
 
Output: 
| Architecture | Training Function (Optimizer) | Accuracy | 
|------------------|----------------------------------------|-------------| 
| [8, 8]        
| [8, 8]       
| [8, 8]        
| [16, 8]      
| [16, 8]       
| [16, 8]       
| Adam                              
| SGD                             
| RMSprop                    
| Adam                               
| SGD                               
| RMSprop                      
| [32, 16, 8]   | Adam                              
| [32, 16, 8]   | SGD                                
| [32, 16, 8]   | RMSprop                           
| 0.9333   | 
| 0.8444   | 
| 0.8889   | 
| 0.7778   | 
| 0.8889   | 
| 0.9333   | 
| 1.0000   | 
| 0.8667   | 
| 1.0000   | 
