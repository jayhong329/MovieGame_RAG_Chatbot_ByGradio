from dotenv import load_dotenv
import os
import openai
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import pymysql

# 載入環境變數
load_dotenv()

# 設定 OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key


# 模型和索引路徑設定
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

index_path = './tmdb_vector.index'
model = SentenceTransformer(model_name)

# 系統提示詞
SYSTEM_PROMPT = """你是一個專業的遊戲推薦助理，擅長根據用戶的提問給出推薦遊戲。
你的任務是：
* 從用戶的問題中，找到重點，並嘗試解析重點
* 基於檢索到的遊戲資訊提供相關推薦
* 解釋推薦的原因
* 提供遊戲的關鍵資訊（類型、導演、劇情簡介等）
* 如果沒有檢索到直接相關的答案，不要亂回答，請嘗試提供同一類型的建議

回答時請：
- 使用正體中文
- 條理清晰地組織信息
- 確保信息準確性
- 必要時提供延伸建議"""


# 資料庫連接配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'P@ssw0rd',  # 替換為您的密碼
    'database': 'tmdb_movie_data',
    'charset': 'utf8mb4'
}

def load_data_from_sql():
    """使用 pymysql 從 SQL 資料庫載入電影數據"""
    try:
        # 創建連接
        connection = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            charset=DB_CONFIG['charset']
        )

        # SQL 查詢語句
        query = "SELECT * FROM movie_top_rated"

        # 使用 Pandas 讀取數據
        data = pd.read_sql(query, con=connection)
        
        # 處理缺失值
        data.fillna('', inplace=True)
        
        print("數據加載成功！")
        return data
    except Exception as e:
        print(f"從 SQL 加載數據時發生錯誤：{str(e)}")
        raise
    finally:
        # 確保連接關閉
        if connection:
            connection.close()

# 替換原代碼
data = load_data_from_sql()


def load_index_and_mappings():
    """載入FAISS索引和遊戲ID映射"""
    index = faiss.read_index(index_path)
    with open('tmdb_ids.pkl', 'rb') as f:
        tmdb_ids = pickle.load(f)
    return index, tmdb_ids

def search_similar_tmdb(query, index, model, top_k=5):
    """搜索相似電影"""
    query_vector = model.encode([query], normalize_embeddings=True)

    # 使用FAISS進行搜索
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    valid_indices = indices[0][indices[0] >= 0]  # 過濾無效索引
    return distances[0][:len(valid_indices)], valid_indices

def get_tmdb_details(tmdb_idx, data):
    """獲取電影詳細信息"""
    try:
        # 確保索引有效
        if 0 <= tmdb_idx < len(data):
            tmdb = data.iloc[tmdb_idx]
            if 'title' in tmdb and tmdb['title']:
                return {
                    'title': tmdb['title'],
                    'overview': tmdb['overview'] or '無電影簡介',
                    'genre_ids': tmdb['genre_ids'] or '未知類型',
                    'vote_average': tmdb['vote_average'] or '未知評分',
                    'release_date': tmdb['release_date'] or '未知日期',
                }
            else:
                print(f"資料缺失或無效，索引：{tmdb_idx}")
        return None  # 索引無效或資料缺失
    except Exception as e:
        print(f"獲取電影詳情錯誤: {str(e)}")
        return None

def get_ai_response(query, retrieved_texts):
    """使用 OpenAI API 生成流式回應"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "system", "content": f"以下是相關電影信息：\n{retrieved_texts}"}
        ]

        # 使用舊版 API 調用方式
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            stream=True
        )

        content = ''
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content'):
                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    content += delta_content
                    yield content

    except Exception as e:
        print(f"AI 處理時發生錯誤：{str(e)}")
        yield f"處理請求時發生錯誤：{str(e)}"

def game_chat(message, history):
    """處理用戶查詢並生成流式回應"""
    try:
        # 檢索相關電影
        index, tmdb_ids = load_index_and_mappings()
        distances, indices = search_similar_tmdb(message, index, model, top_k=3)
        
        # 準備檢索到的電影信息
        similar_tmdb_info = []
        for idx, distance in zip(indices, distances):
            if distance > 0.3:  # 設定相似度閾值
                movie_details = get_tmdb_details(idx, data)
                if movie_details:
                    similar_tmdb_info.append(
                        f"電影名稱：{movie_details['title']}\n"
                        f"電影類型：{movie_details['genre_ids']}\n"
                        f"發行日：{movie_details['release_date']}\n"
                        f"電影簡介：{movie_details['overview']}\n"
                        f"電影評分：{movie_details['vote_average']}\n"
                        f"相似度分數：{distance:.4f}\n"
                    )
                else:
                    print(f"未獲取到有效電影詳情，索引：{idx}")
        
        retrieved_texts = "\n".join(similar_tmdb_info)
        
        # 使用生成器生成流式回應
        last_response = None
        for response_chunk in get_ai_response(message, retrieved_texts):
            last_response = response_chunk
            yield response_chunk

        if last_response is None:
            yield "抱歉，無法生成回應。請稍後再試。"

    except Exception as e:
        print(f"處理查詢時發生錯誤：{str(e)}")
        yield f"無法處理您的請求：{str(e)}"

# 設定介面描述
desc = "直接輸入您想看的電影類型或關鍵詞，AI 助手會為您推薦相關電影。"

article = "<h1>電影推薦系統</h1>"\
         "<h3>使用說明:</h3>"\
         "<ul><li>輸入您感興趣的電影類型、或任何關鍵詞</li>"\
         "<li>系統會根據資料庫內容為您推薦相關電影</li>"\
         "<li>AI 助手會解釋推薦原因並提供電影詳細信息</li></ul>"

demo = gr.ChatInterface(
        fn=game_chat,
        theme="soft",
        title=article,
        description=desc,
        examples=[
            "請推薦幾部動作冒險的電影",
            "經典的科幻類型電影有哪些？",
            "我想看動畫類的電影",
            "有什麼電影是劇情類型的嗎？"
        ]
    )

# 主程序
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)