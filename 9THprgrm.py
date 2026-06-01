import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# Sample Dataset
# Features: [Weight, Texture]
# Texture: 0 = Smooth, 1 = Rough
X = np.array([
    [150, 0],
    [170, 1],
    [120, 0],
    [140, 1],
    [200, 1],
    [130, 0]
])


# 0 = Apple, 1 = Orange
y = np.array([0, 1, 0, 1, 1, 0])


clf = DecisionTreeClassifier(random_state=42)


clf.fit(X, y)


tree_rules = export_text(
    clf,
    feature_names=["Weight", "Texture"]
)

print("Decision Tree Rules:")
print(tree_rules)

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

prediction = clf.predict([[weight, texture]])

if prediction[0] == 0:
    print("Predicted Class: Apple")
else:
    print("Predicted Class: Orange")