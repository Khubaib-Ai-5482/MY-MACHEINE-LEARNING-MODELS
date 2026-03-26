import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("email.csv")

print("Unique Categories:", df["Category"].unique())

le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])

df["Message_length"] = df["Message"].astype(str).apply(len)

plt.figure(figsize=(5,4))
sns.countplot(x="Category", data=df, palette="viridis")
plt.title("Category Distribution (0 = Ham, 1 = Spam)")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data=df, x="Message_length", hue="Category", bins=50, kde=True, palette="Set2")
plt.title("Message Length Distribution by Category")
plt.xlabel("Message Length")
plt.ylabel("Count")
plt.show()

X = df["Message"].astype(str)
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words="english", max_features=5000)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

word_freq = pd.DataFrame(
    X_train_cv.toarray(), columns=cv.get_feature_names_out()
).sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(8,4))
sns.barplot(x=word_freq.values, y=word_freq.index, palette="cubehelix")
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_cv, y_train)

y_pred = model.predict(X_test_cv)

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

