# IMDb 電影評論情緒分析專案

## 一、專案背景與目的

隨著網路評論與使用者生成內容（UGC）的爆炸性成長，企業與平台對於自動判斷評論情緒的需求日益增加。本專案利用機器學習與自然語言處理（NLP）技術，針對 IMDb 電影評論進行情緒分類，協助企業快速理解用戶反饋，提升決策效率。

---

## 二、使用資料集

- **資料來源**：[IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **資料筆數**：共 50,000 筆評論（25,000 筆訓練 + 25,000 筆測試）
- **欄位說明**：
  - `review`：影評文字內容
  - `sentiment`：評論情緒（`positive` 或 `negative`）

---

## 三、使用技術與工具

- **程式語言**：Python
- **主要套件**：
  - `pandas`：資料處理
  - `scikit-learn`：機器學習模型與評估
  - `nltk`：自然語言處理
  - `matplotlib` / `seaborn`：視覺化

- **機器學習模型**：
  - Logistic Regression（邏輯迴歸）

---

## 四、模型流程

### 4.1 資料前處理
- 移除 HTML 標籤與特殊符號
- 將文字轉為小寫
- 過濾英文停用詞（stopwords）
- 斷詞與清洗後生成新欄位 `clean_review`

### 4.2 特徵轉換（TF-IDF 向量化）
- 使用 `TfidfVectorizer` 將文本轉換為向量
- 設定最大特徵數量 `max_features=5000`

### 4.3 模型訓練與預測
- 將資料分割為訓練集與測試集
- 使用邏輯迴歸模型進行訓練
- 對測試集進行預測

### 4.4 模型評估
- 計算準確率（Accuracy）
- 輸出分類報告（Precision、Recall、F1-score）
- 畫出混淆矩陣（Confusion Matrix）視覺化結果

---

## 五、專案成果與結論

- 邏輯迴歸模型在 IMDb 評論測試集上的預測準確率約88%
- 特徵工程（TF-IDF）與基本前處理能有效提升模型效能
- 模型簡單快速，適合作為情緒分析入門範例

---

## 六、未來展望
- 使用深度學習模型如 LSTM 或 BERT 進一步提升準確率
- 擴充應用至中文評論（結合繁體中文 NLP 技術）
- 部署成網頁應用或聊天機器人功能

---

## 此專案使用Chat Gpt完成