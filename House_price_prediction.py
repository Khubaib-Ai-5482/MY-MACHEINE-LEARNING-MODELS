import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("house_price_prediction_.csv")
df = df.drop("id", axis=1)

cat_data = ["location", "has_garage"]
for col in cat_data:
    df[col] = LabelEncoder().fit_transform(df[col])

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

num_cols = ["area_sqft", "bedrooms", "bathrooms", "floors", "year_built"]
plt.figure(figsize=(15,8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=df[col], y=df["price"], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x=df["location"], y=df["price"])
plt.show()

plt.figure(figsize=(6,5))
sns.boxplot(x=df["has_garage"], y=df["price"])
plt.show()

x = df.drop("price", axis=1)
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = r2_score(y_test, y_pred) * 100
print(f"Linear Regression R² Score: {accuracy:.2f}%")

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True)
plt.show()