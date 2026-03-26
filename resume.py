import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("gpt_dataset.csv")
target_job = "Python Developer"
df['label'] = df['Category'].apply(lambda x: 1 if x.strip() == target_job else 0)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['clean_resume'] = df['Resume'].apply(clean_text)

X = df['clean_resume']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['Not Suitable','Suitable'], 
            yticklabels=['Not Suitable','Suitable'], cmap='Blues')
plt.show()

features = tfidf.get_feature_names_out()
coef = model.coef_[0]
top_not_suitable = coef.argsort()[:10]
top_suitable = coef.argsort()[-10:]

plt.barh([features[i] for i in top_not_suitable], coef[top_not_suitable], color='red')
plt.show()
plt.barh([features[i] for i in top_suitable], coef[top_suitable], color='green')
plt.show()

def predict_resume(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)
    return "Suitable" if pred[0]==1 else "Not Suitable"

while True:
    resume_input = input("Enter resume text (or type 'exit' to quit): ")
    if resume_input.lower() == 'exit':
        break
    print("Prediction:", predict_resume(resume_input))
