import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from config import *

# Load data
data = pd.read_csv(protocol_and_attack_csv)

# Display data overview
print("Data Overview:")
print(data.head())

# Extract features
features = data[['SourceIPaddress', 'DestinationIPaddress', 'forward_pkts', 
                 'backward_pkts', 'st_time', 'end_time', 'Source_jitter', 'Destination_jitter', 'protocol', 'service']].copy()

# Combine 'protocol' and 'service' into a single target column
# data['combined_target'] = data['protocol'] + '_' + data['service']

# Extract target
labels = data['attack']

# Convert timestamps to numerical format (Unix timestamp)
features['st_time'] = pd.to_datetime(features['st_time'], unit='s').view(np.int64) // 10**9
features['end_time'] = pd.to_datetime(features['end_time'], unit='s').view(np.int64) // 10**9

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['SourceIPaddress', 'DestinationIPaddress','protocol', 'service'])

# # Encode the combined target column
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels)

# Verify the encoding
print("Feature Head After Encoding:")
print(features.head())

# print("Label Encoding Head:")
# print(pd.DataFrame({'combined_target': labels, 'encoded_label': labels_encoded}).head())

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Dimensionality Reduction with PCA
pca = PCA(n_components=50)  # Adjust number of components if necessary
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# Initialize models
# For KNN with a different distance metric
model2 = KNeighborsClassifier(metric='manhattan')  # Example for Manhattan distance

# For Decision Tree using entropy
# model1 = DecisionTreeClassifier(criterion='entropy')  # Using entropy for splitting


# Train Decision Tree model
# print("Training Decision Tree model...")
# model1.fit(x_train_pca, y_train)
# print("Training completed.")

# Train K-Nearest Neighbors model
print("Training K-Nearest Neighbors model...")
model2.fit(x_train_pca, y_train)
print("Training completed.")

# Predict with both models
print("prediction_start:::::")
# predictionTree = model1.predict(x_test_pca)
# print("prediction_com1:::::")
predictionKNN = model2.predict(x_test_pca)
print("prediction_com2:::::")

# Calculate and print accuracies
# print("Decision Tree Accuracy:", accuracy_score(y_test, predictionTree))
print("KNN Accuracy:", accuracy_score(y_test, predictionKNN))

# Confusion matrices for both models
# cmTree = confusion_matrix(y_test, predictionTree)
cmKNN = confusion_matrix(y_test, predictionKNN)

# print("Confusion Matrix for Decision Tree:")
# print(cmTree)
print("\nConfusion Matrix for KNN:")
print(cmKNN)

# Define a new sample for prediction
new_sample = pd.DataFrame({
    'SourceIPaddress': ['175.45.176.3'],
    'DestinationIPaddress': ['149.171.126.18'],
    'st_time': [1421927415],
    'end_time': [1421927415]
})

# Convert timestamps to numerical format
new_sample['st_time'] = pd.to_datetime(new_sample['st_time'], unit='s').view(np.int64) // 10**9
new_sample['end_time'] = pd.to_datetime(new_sample['end_time'], unit='s').view(np.int64) // 10**9

# One-hot encode categorical features
new_sample = pd.get_dummies(new_sample, columns=['SourceIPaddress', 'DestinationIPaddress'])

# Align new_sample columns with features columns
new_sample = new_sample.reindex(columns=features.columns, fill_value=0)

# Scale and transform the new sample
new_sample_scaled = scaler.transform(new_sample)
new_sample_pca = pca.transform(new_sample_scaled)

# Predict with both models
# new_predictionTree = model1.predict(new_sample_pca)
new_predictionKNN = model2.predict(new_sample_pca)

# Decode the predictions
# decoded_predictionTree = label_encoder.inverse_transform(new_predictionTree)
# decoded_predictionKNN = label_encoder.inverse_transform(new_predictionKNN)

# print("Decision Tree Prediction for New Sample:", decoded_predictionTree)
print("KNN Prediction for New Sample:", new_predictionKNN)
