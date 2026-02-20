import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("StudentsPerformance.csv")

# =========================
# 2. Feature engineering
# =========================
# Create overall average score
df["average_score"] = df[
    ["math score", "reading score", "writing score"]
].mean(axis=1)

# =========================
# 3. Filter target group
# Parents without higher education
# =========================
target_group = df[
    df["parental level of education"].isin(
        ["some high school", "high school"]
    )
]

# =========================
# 4. Split into two groups
# =========================
completed = target_group[
    target_group["test preparation course"] == "completed"
]

not_completed = target_group[
    target_group["test preparation course"] == "none"
]

print("Sample sizes:")
print(f"Completed course: {len(completed)} students")
print(f"No course: {len(not_completed)} students\n")

# =========================
# 5. Calculate statistics
# =========================
completed_mean = completed["average_score"].mean()
not_completed_mean = not_completed["average_score"].mean()

print("Average Scores:")
print(f"Completed course: {completed_mean:.2f}")
print(f"No course: {not_completed_mean:.2f}\n")

# =========================
# 6. Statistical test
# =========================
t_stat, p_value = ttest_ind(
    completed["average_score"],
    not_completed["average_score"],
    equal_var=False
)

print("T-test results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}\n")

if p_value < 0.05:
    print("Result: The difference is statistically significant.")
else:
    print("Result: The difference is NOT statistically significant.")

# =========================
# 7. Visualization
# =========================
plt.figure(figsize=(8, 6))

plt.boxplot(
    [
        completed["average_score"],
        not_completed["average_score"],
    ],
    labels=["Completed Course", "No Course"],
)

plt.title("Impact of Test Preparation Course on Exam Performance")
plt.ylabel("Average Exam Score")

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
