import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print(df.head())
print(df['label'].value_counts())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

df['label'].value_counts().plot(kind='bar')
plt.title("Class Distribution (Ham vs Spam)")
plt.xticks([0,1], ['Ham', 'Spam'], rotation=0)
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

df['text_length'] = df['text'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(df[df['label']==0]['text_length'], bins=40, label='Ham')
sns.histplot(df[df['label']==1]['text_length'], bins=40, label='Spam')
plt.legend()
plt.title("Text Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

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
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_names = tfidf.get_feature_names_out()
spam_prob = model.feature_log_prob_[1]

top_spam_words = np.argsort(spam_prob)[-15:]

plt.figure(figsize=(8,4))
plt.barh([feature_names[i] for i in top_spam_words], spam_prob[top_spam_words])
plt.title("Top Spam Indicative Words")
plt.xlabel("Log Probability")
plt.show()

def predict_spam(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)
    return "Spam" if pred[0] == 1 else "Ham"

print(predict_spam("Congratulations! You have won a free iPhone"))
print(predict_spam("Kal class kis time hai?"))
