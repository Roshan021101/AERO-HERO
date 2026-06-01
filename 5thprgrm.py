import pandas as pd
import matplotlib.pyplot as plt

# Dataset of 13 Students
data = {
    "Student_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 7, 9, 10],
    "Exam_Score": [35, 40, 45, 50, 60, 65, 70, 80, 42, 62, 75, 88, 95]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df["Study_Hours"], df["Exam_Score"], s=100)

# Display Student IDs
for i in range(len(df)):
    plt.text(
        df["Study_Hours"][i],
        df["Exam_Score"][i],
        str(df["Student_ID"][i])
    )

plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

# Even-number study hours on X-axis
plt.xticks(range(0, 12, 2))

plt.grid(True)
plt.show()