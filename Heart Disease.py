import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("heart.csv")

print(df.shape)

x = df.drop("target",axis=1)
y = df["target"]

plt.figure(figsize=(5,4))
sns.countplot(x="target",data=df,palette="Set2")
plt.title("Target Distribution (0 = Healthy, 1 = Heart Disease)")
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)*100
print("Accuracy:",accuracy)
print("Classification Report:")
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feat_imp = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Feature Importance - RandomForest")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=20, kde=True, color='orange')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x="age", y="chol", hue="target", data=df, palette="coolwarm")
plt.title("Age vs Cholesterol (by Target)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=False, cmap="crest")
plt.title("Feature Correlation Heatmap")
plt.show()


