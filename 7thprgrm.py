import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset: [Maths Marks, Science Marks]
X = np.array([
    [35, 40],
    [45, 50],
    [20, 25],
    [80, 85],
    [75, 70],
    [30, 35],
    [90, 95],
    [60, 65],
    [25, 30],
    [85, 80]
])

# Target: 0 = Fail, 1 = Pass
y = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Predicted Values:", y_pred)
print("Actual Values:", y_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict for a new student
maths = int(input("\nEnter Maths Marks: "))
science = int(input("Enter Science Marks: "))

prediction = knn.predict([[maths, science]])

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")