import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("Gold_price_2025 new.csv")

cat_data = df.select_dtypes(include="object")
for col in cat_data:
    df[col] = LabelEncoder().fit_transform(df[col])

x = df.drop("priceHigh", axis=1)
y = df["priceHigh"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = r2_score(y_test, y_pred) * 100
print(f"Decision Tree R² Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Gold Price High")
plt.legend()
plt.grid(True)
plt.show()

