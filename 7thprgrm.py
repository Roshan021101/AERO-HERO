# WAP to implement k nearest neighbour classifier using scikit learn and train the classifier o n the dataset 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = np.array([
    [35,40],[45,50],[20,25],[80,85],[75,70],
    [30,35],[90,95],[60,65],[25,30],[85,80]
])

y = np.array([0,1,0,1,1,0,1,1,0,1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Predicted Values:", y_pred)
print("Actual Values:", y_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

maths = int(input("\nEnter Maths Marks: "))
science = int(input("Enter Science Marks: "))

print("Result: PASS" if knn.predict([[maths, science]])[0] else "Result: FAIL")
