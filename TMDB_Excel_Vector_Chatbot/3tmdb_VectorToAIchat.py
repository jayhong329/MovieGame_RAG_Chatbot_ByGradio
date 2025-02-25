from dotenv import load_dotenv
import os
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from openai import OpenAI

# 載入環境變數
load_dotenv()

# 設定 OpenAI API (新 - OpenAI )
openai_api_key = os.getenv('OPENAI_API_KEY')
OpenAI.api_key = openai_api_key


# 模型和索引路徑設定
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

index_path = './tmdb_excel_vector.index'
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


def load_data_from_excel(file_path):
    """從 Excel 文件讀取電影數據"""
    try:
        print(f"從 {file_path} 讀取電影數據...")
        data = pd.read_excel(file_path)

        # 检查是否有数据
        if data.empty:
            raise ValueError(f"Excel 文件 {file_path} 中沒有數據。")
        
        # 填补空值
        data['movie_description'] = data['movie_description'].fillna('')

        print("Excel 數據讀取完成。")
        return data
    except Exception as e:
        print(f"讀取 Excel 文件時發生錯誤：{str(e)}")
        raise e

# 修正：提供 Excel 檔案路徑
data = load_data_from_excel('../tmdb_top_rated_movies_total.xlsx')  # Excel 檔案路徑


def load_index_and_mappings():
    """載入FAISS索引和遊戲ID映射"""
    index = faiss.read_index(index_path)
    with open('tmdb_excel_ids.pkl', 'rb') as f:
        tmdb_excel_ids = pickle.load(f)

    print(f"載入 ID 映射，共 {len(tmdb_excel_ids)} 條數據")
    print(f"ID 映射前 10 筆數據: {list(tmdb_excel_ids.items())[:10]}")
    return index, tmdb_excel_ids

def search_similar_tmdb(query, index, model, top_k=5):
    """搜索相似電影"""
    query_vector = model.encode([query], normalize_embeddings=True)

    # 使用FAISS進行搜索
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    valid_indices = indices[0][indices[0] >= 0]  # 過濾無效索引
    return distances[0][:len(valid_indices)], valid_indices

def get_tmdb_details(tmdb_idx, data, id_mapping):
    """獲取電影詳細信息"""
    try:
        if tmdb_idx in id_mapping:  # 檢查索引是否存在於 ID 映射
            df_idx = id_mapping[tmdb_idx]  # 獲取對應的 DataFrame 行號
            
            # 確保 df_idx 是整數並且在有效範圍內
            if isinstance(df_idx, int) and 0 <= df_idx < len(data):
                tmdb = data.iloc[df_idx]

                # 添加調試信息
                print(f"處理索引 {tmdb_idx}:")
                print(f"- DataFrame 索引: {df_idx}")
                print(f"- 電影標題: {tmdb.get('movie_title', 'N/A')}")

                return {
                    'title': str(tmdb.get('movie_title', '')),
                    'overview': str(tmdb.get('movie_description', '')) or '無電影簡介',
                    'genre_ids': str(tmdb.get('movie_genre', '')) or '未知類型',
                    'vote_average': float(tmdb.get('vote_average', 0)) or 0.0,
                    'release_date': str(tmdb.get('release_date', '')) or '未知日期',
                }
        print(f"索引 {tmdb_idx} 無效或無對應數據")
        return None  # 索引無效或資料缺失
    except Exception as e:
        print(f"獲取電影詳情錯誤: {str(e)}")
        print(f"相關資訊：tmdb_idx={tmdb_idx}, df_idx類型={type(df_idx)}")
        return None

def get_ai_response(query, retrieved_texts):
    """使用 OpenAI API 生成流式回應"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "system", "content": f"以下是相關電影信息：\n{retrieved_texts}"}
        ]

        client = OpenAI()

        # 使用新版 API 調用方式
        response = client.chat.completions.create(
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
        # 將歷史記錄納入上下文
        history_context = "\n".join([f"用戶：{h[0]}\n助理：{h[1]}" for h in history if len(h) == 2])

        # 檢索相關電影
        index, tmdb_ids = load_index_and_mappings()
        print(f"FAISS 索引總數: {index.ntotal}")
        distances, indices = search_similar_tmdb(message, index, model, top_k=3)
        print(f"檢索返回索引: {indices}")
        print(f"檢索返回距離: {distances}")
        
        # 準備檢索到的電影信息
        similar_tmdb_info = []
        for idx, distance in zip(indices, distances):
            print(f"\n處理檢索結果：索引={idx}, 距離={distance}")
            print(f"索引是否在映射中: {idx in tmdb_ids}")

            if distance > 0.3:  # 設定相似度閾值
                movie_details = get_tmdb_details(idx, data, tmdb_ids)  # 傳遞 id_mapping
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

        # 構造 OpenAI 提示詞，加入歷史記錄
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"以下是歷史對話：\n{history_context}"} if history_context else None,
            {"role": "user", "content": message},
            {"role": "system", "content": f"以下是相關電影信息：\n{retrieved_texts}"}
        ]
        messages = [m for m in messages if m]  # 過濾空值
        
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