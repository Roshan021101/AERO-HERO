# write a program to visualize the dataset to gain insights by using matplotlib by plotting scatterplot

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Student_ID": range(1, 14),
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 7, 9, 10],
    "Exam_Score": [35, 40, 45, 50, 60, 65, 70, 80, 42, 62, 75, 88, 95]
})

print("Student Dataset:")
print(df)

plt.figure(figsize=(8, 6))
plt.scatter(df["Study_Hours"], df["Exam_Score"], s=100)

for x, y, sid in zip(df["Study_Hours"], df["Exam_Score"], df["Student_ID"]):
    plt.text(x, y, sid)

plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.xticks(range(0, 12, 2))
plt.grid(True)
plt.show()
