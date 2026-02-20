import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# =========================
# 1. Load data
# =========================
data = pd.read_csv("DataAnalyst.csv")

# =========================
# 2. Salary extraction
# =========================
def extract_salary(s):
    match = re.search(r"\$(\d+)[Kk]?\s*-\s*\$(\d+)[Kk]?", str(s))
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2 * 1000
    return None

data["Average Salary"] = data["Salary Estimate"].apply(extract_salary)

# Drop rows without salary
data = data.dropna(subset=["Average Salary"])

# =========================
# 3. Skills extraction
# =========================
skills = ["SQL", "Python", "R", "Excel", "Tableau", "Power BI", 
          "SAS", "Statistics", "Machine Learning"]

for skill in skills:
    data[skill] = data["Job Description"].str.contains(skill, case=False, na=False)

# =========================
# 4. Salary comparison per skill
# =========================
salary_by_skill = {}

for skill in skills:
    mean_salary = data.groupby(skill)["Average Salary"].mean()
    salary_by_skill[skill] = mean_salary.get(True, 0)

salary_df = pd.Series(salary_by_skill).sort_values(ascending=False)

print("Average salary by skill:")
print(salary_df)

# =========================
# 5. Regression model
# =========================
X = data[skills]
y = data["Average Salary"]

model = LinearRegression()
model.fit(X, y)

coef_df = pd.Series(model.coef_, index=skills).sort_values(ascending=False)

print("\nRegression coefficients (impact on salary):")
print(coef_df)

# =========================
# 6. Visualization
# =========================
plt.figure(figsize=(10,6))
salary_df.plot(kind="bar")
plt.title("Average Salary by Skill")
plt.ylabel("Salary ($)")
plt.show()

# =========================
# 7. Skill count impact
# =========================
data["Skill Count"] = data[skills].sum(axis=1)

plt.figure(figsize=(8,6))
sns.scatterplot(x="Skill Count", y="Average Salary", data=data)
plt.title("Number of Skills vs Salary")
plt.show()
