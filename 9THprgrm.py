import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

X = np.array([
    [150,0], [170,1], [120,0],
    [140,1], [200,1], [130,0]
])

y = np.array([0, 1, 0, 1, 1, 0])

clf = DecisionTreeClassifier(random_state=42).fit(X, y)

print("Decision Tree Rules:")
print(export_text(clf, feature_names=["Weight", "Texture"]))

plt.figure(figsize=(10, 6))
plot_tree(
    clf,
    feature_names=["Weight", "Texture"],
    class_names=["Apple", "Orange"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Classifier")
plt.show()

weight = float(input("Enter fruit weight: "))
texture = int(input("Enter texture (0=Smooth, 1=Rough): "))

print(
    "Predicted Class: Apple"
    if clf.predict([[weight, texture]])[0] == 0
    else "Predicted Class: Orange"
)
