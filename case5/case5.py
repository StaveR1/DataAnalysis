import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency

# =========================
# 1. Load data
# =========================
df = pd.read_csv("Space_Corrected.csv")
df.columns = df.columns.str.strip()

df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce', utc=True)
df = df.dropna(subset=['Datum', 'Status Mission'])

df['Success'] = df['Status Mission'] == 'Success'

# =========================
# 2. Feature Engineering
# =========================
df['Hour'] = df['Datum'].dt.hour
df['Year'] = df['Datum'].dt.year
df['Morning'] = df['Hour'].between(5, 11)

df['Private'] = ~df['Company Name'].isin(
    ['Roscosmos', 'CASC', 'VKS RF', 'JAXA', 'MHI', 'IAI']
)

# =========================
# 3. Baseline success rate
# =========================
print("Overall success rate:", df['Success'].mean())

# =========================
# 4. Hypothesis test (Chi-square)
# =========================
contingency = pd.crosstab(
    df[df['Morning']]['Private'],
    df[df['Morning']]['Success']
)

chi2, p, _, _ = chi2_contingency(contingency)
print("Chi-square p-value (Morning launches):", p)

# =========================
# 5. ML Model
# =========================
features = ['Company Name', 'Location', 'Rocket', 'Hour', 'Year', 'Private']
X = df[features]
y = df['Success']

categorical = ['Company Name', 'Location', 'Rocket']
numeric = ['Hour', 'Year', 'Private']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
