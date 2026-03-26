import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("IRIS.csv")

cat_data = df.select_dtypes(include="object")
le = LabelEncoder()
for col in cat_data:
    df[col] = le.fit_transform(df[col])

x = df.drop("species", axis=1)
y = df["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.plot(df["sepal_length"])
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.grid()
plt.show()

sns.histplot(df["petal_width"], bins=20, kde=True, color="skyblue")
plt.xlabel("Petal Width")
plt.ylabel("Frequency")
plt.grid()
plt.show()

sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


