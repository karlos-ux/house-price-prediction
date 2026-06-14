import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv('data/housing.csv')

# -------------------------
# Basic cleaning
# -------------------------
# Remove missing values (important)
df = df.dropna()

# -------------------------
# Convert categorical → numeric
# -------------------------
df = pd.get_dummies(df, drop_first=True)

# -------------------------
# Features & Target
# -------------------------
if 'price' not in df.columns:
    raise ValueError("❌ 'price' column not found in dataset")

X = df.drop('price', axis=1)
y = df['price']

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Save model
# -------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# -------------------------
# Save column structure (VERY IMPORTANT)
# -------------------------
with open('columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and columns saved successfully!")