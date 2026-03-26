import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("spotify_history.csv") 

df["reason_start"] = df["reason_start"].fillna(df["reason_start"].mode()[0])
df["reason_end"] = df["reason_end"].fillna(df["reason_end"].mode()[0])

df = df.drop(['spotify_track_uri', 'ts', 'track_name', 'artist_name', 'album_name'], axis=1)

cat_data = df.select_dtypes(include="object")
for col in cat_data:
    df[col] = LabelEncoder().fit_transform(df[col])

x = df.drop("skipped", axis=1)
y = df["skipped"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_ = model.predict(x_train)
accuracy = accuracy_score(y_test, y_pred) * 100
accuracy_ = accuracy_score(y_train, y_pred_) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
# print(classification_report(y_test, y_pred))
print(accuracy_)
conf_mat = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Skipped", "Skipped"], yticklabels=["Not Skipped", "Skipped"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# plt.figure(figsize=(5,4))
# sns.barplot(x=["Accuracy"], y=[accuracy], palette="pastel")
# plt.ylim(0, 100)
# plt.ylabel("Accuracy %")
# plt.show()


