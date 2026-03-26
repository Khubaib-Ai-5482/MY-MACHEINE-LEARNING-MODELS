import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("15_fake_news_detection.csv")
df = df.dropna(subset=['title','text','label'])
df['label'] = df['label'].str.upper().str.strip()
df['label'] = df['label'].map({'FAKE':0,'REAL':1})
df = df.dropna(subset=['label'])
df['full_text'] = df['title'] + " " + df['text']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_text'] = df['full_text'].apply(clean_text)

plt.figure(figsize=(5,4))
sns.barplot(x=df['label'].value_counts().index, y=df['label'].value_counts().values, palette='viridis')
plt.xticks([0,1],['Fake','Real'])
plt.ylabel("Count")
plt.show()

df['text_len'] = df['full_text'].apply(len)
plt.figure(figsize=(6,4))
sns.histplot(df[df['label']==0]['text_len'], bins=50, color='red', label='Fake', alpha=0.7)
sns.histplot(df[df['label']==1]['text_len'], bins=50, color='green', label='Real', alpha=0.7)
plt.legend(title="Label")
plt.show()

X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", round(model.score(X_test_tfidf, y_test)*100,2), "%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'], cmap='Blues')
plt.show()

features = tfidf.get_feature_names_out()
coef = model.coef_[0]

top_fake = coef.argsort()[:10]
top_real = coef.argsort()[-10:]

plt.figure(figsize=(7,4))
plt.barh([features[i] for i in top_fake], coef[top_fake], color='red')
plt.show()

plt.figure(figsize=(7,4))
plt.barh([features[i] for i in top_real], coef[top_real], color='green')
plt.show()

def predict_news(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)
    return "REAL NEWS" if pred[0]==1 else "FAKE NEWS"

while True:
    text = input("\nType news text (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    print("Prediction:", predict_news(text))
