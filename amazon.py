import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("amazon.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['Text'].apply(clean_text)

plt.figure(figsize=(5,4))
sns.barplot(x=df['label'].value_counts().index, y=df['label'].value_counts().values, palette='viridis')
plt.xticks([0,1],['Negative','Positive'])
plt.ylabel("Count")
plt.show()

X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'], cmap='Blues')
plt.show()

features = tfidf.get_feature_names_out()
coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]  # difference between positive & negative

top_negative = coef.argsort()[:10]
top_positive = coef.argsort()[-10:]

plt.figure(figsize=(7,4))
plt.barh([features[i] for i in top_negative], coef[top_negative], color='red')
plt.title("Top words indicating Negative reviews")
plt.show()

plt.figure(figsize=(7,4))
plt.barh([features[i] for i in top_positive], coef[top_positive], color='green')
plt.title("Top words indicating Positive reviews")
plt.show()

def predict_review(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)
    return "Positive" if pred[0]==1 else "Negative"

print(predict_review("Amazing product, works perfectly!"))
print(predict_review("Very poor quality, broke in a week"))
