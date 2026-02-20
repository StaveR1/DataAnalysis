import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# =========================
# 1. Load data
# =========================
df = pd.read_csv("countries of the world.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns for convenience
df = df.rename(columns={
    "GDP ($ per capita)": "GDP",
    "Literacy (%)": "Literacy"
})

# =========================
# 2. Data cleaning
# =========================
for col in ["GDP", "Literacy"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove missing values
df = df.dropna(subset=["GDP", "Literacy"])

# =========================
# 3. Correlation
# =========================
corr, p_value = pearsonr(df["Literacy"], df["GDP"])

print("Pearson correlation:", round(corr, 4))
print("P-value:", round(p_value, 6))

# =========================
# 4. Linear regression
# =========================
X = df[["Literacy"]]
y = df["GDP"]

model = LinearRegression()
model.fit(X, y)

r_squared = model.score(X, y)
print("R-squared:", round(r_squared, 4))

# =========================
# 5. Visualization
# =========================
plt.figure(figsize=(8, 6))
sns.regplot(x="Literacy", y="GDP", data=df)

plt.title("GDP vs Literacy Rate")
plt.xlabel("Literacy Rate (%)")
plt.ylabel("GDP per capita ($)")
plt.show()
