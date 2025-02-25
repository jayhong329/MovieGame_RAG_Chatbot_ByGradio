from dotenv import load_dotenv
import os
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from openai import OpenAI

# 取得當前工作目錄 (cwd)
cwd = os.getcwd()

# 載入環境變數
load_dotenv()

# 設定 OpenAI API (新 - OpenAI )
openai_api_key = os.getenv('OPENAI_API_KEY')
OpenAI.api_key = openai_api_key


# 模型和索引路徑設定
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

model = SentenceTransformer(model_name)

# 檔案路徑
MOVIES_EXCEL = os.path.join(cwd, "original_data/dataMovie.xlsx")
GAMES_EXCEL = os.path.join(cwd, "original_data/dataGame.xlsx")
MOVIES_INDEX_PATH = os.path.join(cwd, "vector_data/movies_excel_vector.index")
GAMES_INDEX_PATH = os.path.join(cwd, "vector_data/games_excel_vector.index")
MOVIES_IDS_PATH = os.path.join(cwd, "vector_data/movies_excel_ids.pkl")
GAMES_IDS_PATH = os.path.join(cwd, "vector_data/games_excel_ids.pkl")

# 系統提示詞
SYSTEM_PROMPT = """你是一個專業且熟悉"電影、遊戲"的查找和推薦助理，擅長根據用戶的提問給出推薦的電影、遊戲。
你的任務是：
* 從用戶的問題中，找到重點，並嘗試解析重點
* 基於檢索到的電影、遊戲資訊（名稱、類型、電影簡介、遊戲簡介等）提供相關推薦，並解釋推薦的原因
* 用戶提問的內容可能包含電影、遊戲名稱、類型、關鍵詞等，除非用戶要求同時提供電影和遊戲的信息，否則只需回答其中一種
* 如果沒有檢索到直接相關的答案，不要亂回答，請嘗試提供同一類型的建議
* 

回答時請：
- 使用正體中文
- 條理清晰地組織信息
- 確保信息準確性
- 必要時提供延伸建議"""


def load_data_from_excel(file_path, is_movie=True):
    """從 Excel 文件讀取數據"""
    try:
        print(f"從 {file_path} 讀取數據...")
        data = pd.read_excel(file_path)

        # 检查是否有数据
        if data.empty:
            raise ValueError(f"Excel 文件 {file_path} 中沒有數據。")
        
        # 根據數據類型刪除沒有簡介的資料
        if is_movie:
            print("刪除沒有電影簡介的資料...")
            data = data.dropna(subset=['movie_description'])
        else:
            print("刪除沒有遊戲簡介的資料...")
            data = data.dropna(subset=['game_description'])

        # 確認是否還有足夠的資料
        if data.empty:
            raise ValueError(f"刪除空值後沒有剩餘資料")
        else:
            print(f"處理後剩餘資料筆數：{len(data)}")

        print("Excel 數據讀取完成。")
        return data
    except Exception as e:
        print(f"讀取 Excel 文件時發生錯誤：{str(e)}")
        raise e

# 分別讀取電影和遊戲數據，確保與向量索引使用相同的資料
movies_data = load_data_from_excel(MOVIES_EXCEL, is_movie=True)
games_data = load_data_from_excel(GAMES_EXCEL, is_movie=False)

def load_index_and_mappings():
    """載入FAISS索引和電影.遊戲ID映射"""
    try:
        # 載入電影索引和映射
        movies_index = faiss.read_index(MOVIES_INDEX_PATH)
        with open(MOVIES_IDS_PATH, 'rb') as f:
            movies_ids = pickle.load(f)
            
        # 載入遊戲索引和映射
        games_index = faiss.read_index(GAMES_INDEX_PATH)
        with open(GAMES_IDS_PATH, 'rb') as f:
            games_ids = pickle.load(f)
            
        return movies_index, movies_ids, games_index, games_ids
    except Exception as e:
        print(f"載入索引時發生錯誤：{str(e)}")
        raise e

def search_similar_items(query, index, model, top_k=5):
    """搜索相似項目（電影或遊戲）"""
    query_vector = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    valid_indices = indices[0][indices[0] >= 0]
    return distances[0][:len(valid_indices)], valid_indices

def get_item_details(idx, data, is_movie=True):
    """獲取項目詳細信息（電影或遊戲）"""
    try:
        if 0 <= idx < len(data):
            item = data.iloc[idx]
            if is_movie:
                return {
                    'movie_title': item['movie_title'],
                    'movie_description': item['movie_description'] or '無電影簡介',
                    'movie_genre': item['movie_genre'] or '未知類型',
                    'vote_average': item['vote_average'] or '未知評分',
                    'release_date': item['release_date'] or '未知日期',
                    'type': '電影'
                }
            else:
                return {
                    'game_title': item['game_title'],
                    'game_description': item['game_description'] or '無遊戲簡介',
                    'game_genre': item['game_genre'] or '未知類型',
                    'game_production': item['game_production'] or '未知製作商',
                    'release_date': item['release_date'] or '未知日期',
                    'type': '遊戲'
                }
        return None
    except Exception as e:
        print(f"獲取詳情錯誤: {str(e)}")
        return None

def get_ai_response(query, retrieved_texts):
    """使用 OpenAI API 生成流式回應"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "system", "content": f"以下是相關信息：\n{retrieved_texts}"}
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

def user_chat(message, history):
    """處理用戶查詢並生成流式回應"""
    try:
        history_context = "\n".join([f"用戶：{h[0]}\n助理：{h[1]}" for h in history if len(h) == 2])

        # 載入索引和映射
        movies_index, movies_ids, games_index, games_ids = load_index_and_mappings()
        
        # 搜索相似電影和遊戲
        movie_distances, movie_indices = search_similar_items(message, movies_index, model, top_k=3)
        game_distances, game_indices = search_similar_items(message, games_index, model, top_k=3)
        
        # 準備檢索到的信息
        similar_items_info = []
        
        # 處理電影結果
        for idx, distance in zip(movie_indices, movie_distances):
            if distance > 0.3:
                movie_details = get_item_details(idx, movies_data, is_movie=True)
                if movie_details:
                    similar_items_info.append(
                        f"類型：電影\n"
                        f"名稱：{movie_details['movie_title']}\n"
                        f"類型：{movie_details['movie_genre']}\n"
                        f"發行日：{movie_details['release_date']}\n"
                        f"簡介：{movie_details['movie_description']}\n"
                        f"評分：{movie_details['vote_average']}\n"
                        f"相似度分數：{distance:.4f}\n"
                    )
        
        # 處理遊戲結果
        for idx, distance in zip(game_indices, game_distances):
            if distance > 0.3:
                game_details = get_item_details(idx, games_data, is_movie=False)
                if game_details:
                    similar_items_info.append(
                        f"類型：遊戲\n"
                        f"名稱：{game_details['game_title']}\n"
                        f"類型：{game_details['game_genre']}\n"
                        f"發行日：{game_details['release_date']}\n"
                        f"簡介：{game_details['game_description']}\n"
                        f"遊戲製造商：{game_details['game_production']}\n"
                        f"相似度分數：{distance:.4f}\n"
                    )
        
        retrieved_texts = "\n".join(similar_items_info)

        # 生成回應
        for response_chunk in get_ai_response(message, retrieved_texts):
            yield response_chunk

    except Exception as e:
        print(f"處理查詢時發生錯誤：{str(e)}")
        yield f"無法處理您的請求：{str(e)}"

# 設定介面描述
desc = "直接輸入您想看的電影或遊戲類型、關鍵詞，AI 助手會為您推薦相關內容。"

article = "<h1>電影與遊戲推薦系統</h1>"\
         "<h3>使用說明:</h3>"\
         "<ul><li>輸入您感興趣的類型或關鍵詞</li>"\
         "<li>系統會推薦相關電影和遊戲</li>"\
         "<li>AI 助手會解釋推薦原因並提供詳細信息</li></ul>"

demo = gr.ChatInterface(
        fn=user_chat,
        theme="soft",
        title=article,
        description=desc,
        examples=[
            "推薦幾部動作、冒險的電影",
            "經典的科幻類型電影有哪些？",
            "推薦動畫類的電影",
            "有什麼電影是劇情類型的嗎？",
            "跟三國志類似的遊戲有哪些？",
            "我想玩角色扮演類的遊戲，推薦幾款給我",
        ]
    )

# 主程序
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)