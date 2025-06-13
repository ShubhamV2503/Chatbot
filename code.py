# A simple AI program using scikit-learn to classify data with a Decision Tree
# Generated on 2025-06-13T03:30:05.593Z

# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Generate a synthetic dataset for classification
X, y = make_classification(
    n_samples=1000,           # Number of data points
    n_features=5,             # Number of features
    n_informative=3,          # Number of informative features
    n_redundant=0,            # Number of redundant features
    random_state=255  # Random seed for reproducibility
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=6)  # Random max_depth between 3 and 7

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Display a few predictions
print("Sample Predictions (first 5 test samples):")
for i in range(5):
    print(f"Sample {i+1}: Features={X_test[i]}, Predicted Class={y_pred[i]}, Actual Class={y_test[i]}")

# Feature importance (how much each feature contributes to the decision)
print("Feature Importance:", clf.feature_importances_)