import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. 載入資料集
df = pd.read_csv("IMDB Dataset.csv")

# 2. 資料前處理
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # 移除HTML標籤
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # 移除非字母
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

# 3. 標籤轉換（positive → 1, negative → 0）
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 4. 分割訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
)

# 5. 特徵轉換（TF-IDF）
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. 模型訓練（Logistic Regression）
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. 預測與評估
y_pred = model.predict(X_test_vec)

print("準確率：", accuracy_score(y_test, y_pred))
print("\n分類報告：\n", classification_report(y_test, y_pred))

# 8. 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('預測')
plt.ylabel('實際')
plt.title('混淆矩陣')
plt.tight_layout()
plt.show()

# 9. 測試單句評論
sample_review = "This movie was absolutely fantastic and thrilling!"
sample_clean = clean_text(sample_review)
sample_vec = vectorizer.transform([sample_clean])
sample_pred = model.predict(sample_vec)
print("\n測試評論：", sample_review)
print("情緒預測結果：", "正面" if sample_pred[0] == 1 else "負面")