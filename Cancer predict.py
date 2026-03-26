import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("cancer_dataset.csv")
df = df.drop("id", axis=1)
df = df.drop("Unnamed: 32", axis=1)

cat_data = df.select_dtypes(include="object")
le = LabelEncoder()
for col in cat_data:
    df[col] = le.fit_transform(df[col])

x = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

sns.countplot(data=df, x="diagnosis")
plt.xlabel("Diagnosis (0 = Benign, 1 = Malignant)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

num_features = df.drop("diagnosis", axis=1).columns
df[num_features].hist(figsize=(15,12), bins=20)
plt.show()

sns.pairplot(df[['diagnosis'] + list(num_features[:5])], hue='diagnosis', palette="Set1")
plt.show()

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances = feat_importances.sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()