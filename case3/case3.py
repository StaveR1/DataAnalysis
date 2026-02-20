import pandas as pd
import matplotlib.pyplot as plt
import itertools

# =========================
# 1. Load data
# =========================
data = pd.read_csv("menu.csv")

# Remove missing values
data = data.dropna()

# =========================
# 2. Define healthy limits
# Based on general dietary guidelines
# =========================
CALORIES_LIMIT = 600
FAT_LIMIT = 20
SODIUM_LIMIT = 800

# =========================
# 3. Filter healthy single items
# =========================
healthy_items = data[
    (data["Calories"] <= CALORIES_LIMIT) &
    (data["Total Fat"] <= FAT_LIMIT) &
    (data["Sodium"] <= SODIUM_LIMIT)
]

print(f"Number of healthy single items: {len(healthy_items)}")

# =========================
# 4. Breakfast analysis
# =========================
breakfast = data[data["Category"] == "Breakfast"]

healthy_breakfast = breakfast[
    (breakfast["Calories"] <= CALORIES_LIMIT) &
    (breakfast["Total Fat"] <= FAT_LIMIT) &
    (breakfast["Sodium"] <= SODIUM_LIMIT)
]

if not healthy_breakfast.empty:
    print("Healthy breakfast options exist.")
else:
    print("Healthy breakfast options are limited.")

# =========================
# 5. Find healthy 2-item combinations
# =========================
breakfast_list = breakfast.to_dict("records")

valid_combinations = []

for combo in itertools.combinations(breakfast_list, 2):
    total_calories = combo[0]["Calories"] + combo[1]["Calories"]
    total_fat = combo[0]["Total Fat"] + combo[1]["Total Fat"]
    total_sodium = combo[0]["Sodium"] + combo[1]["Sodium"]

    if (total_calories <= CALORIES_LIMIT and
        total_fat <= FAT_LIMIT and
        total_sodium <= SODIUM_LIMIT):
        
        valid_combinations.append(
            (combo[0]["Item"], combo[1]["Item"])
        )

print(f"Number of healthy breakfast combinations: {len(valid_combinations)}")

# =========================
# 6. Visualization
# =========================
plt.figure(figsize=(8, 6))
plt.hist(data["Calories"], bins=25)
plt.title("Calories Distribution in McDonald's Menu")
plt.xlabel("Calories")
plt.ylabel("Number of Items")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data["Total Fat"], data["Calories"], alpha=0.5)
plt.title("Fat vs Calories")
plt.xlabel("Total Fat (g)")
plt.ylabel("Calories")
plt.show()
