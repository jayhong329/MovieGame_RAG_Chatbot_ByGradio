Movie&Game ChatBot - with OpenAI_API By Gradio
---

### 訓練資料來源
- The Movie Database (TMDB) (https://www.themoviedb.org/)

### 基礎模型
- [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-similarity-models)

### 安裝套件
- gradio          5.6.0
- gradio_client   1.4.3
- jieba           0.42.1
- fuzzywuzzy      0.18.0
- faiss           1.8.0
- rank-bm25       0.2.2
- pickleshare     0.7.5
- openai          1.64.0
- sentence-transformers   3.1.1

### 說明
- 使用 Sentence-BERT - paraphrase-multilingual-MiniLM-L12-v2  (用來創建向量索引  保存 ID 映射)
- 查詢前處理 (結巴斷詞) + FuzzyWuzzy 進行模糊比對 + BM25 + One-Hot Encoding 類型檢索 + FAISS + Sentence-BERT 語義檢索

- 總共分以下三個檔案夾:
* 一. TMDB_SQL_Vector_Chatbot
--- 透過 TMDB Api 抓取資料，存入 SQL 內，生成向量索引，使用餘弦相似度 cosine similarity ，構建 Gradio_ChatBot
* 二. TMDB_Excel_Vector_Chatbot
--- 透過 TMDB Api 抓取資料，存成 EXCEL ，生成向量索引，使用餘弦相似度 cosine similarity ，構建 Gradio_ChatBot
* 三. CombineFile
--- 使用 Movies & Games 兩份檔案，創建向量索引，透過 Hybrid_search ( FuzzyWuzzy + BM25 + FAISS) ，構建 Gradio_ChatBot

### 成果
- 執行過程的擷圖
![chatbot2](https://github.com/user-attachments/assets/5af6d7ca-1a27-4464-9eeb-46aba74829a9)
![chatbot1](https://github.com/user-attachments/assets/e67f1324-2e1c-42a1-9a56-0035a6e6e837)

- [電影遊戲Chatbot]
https://github.com/user-attachments/assets/059af710-ca49-49f5-8a55-fc895319c9b0

