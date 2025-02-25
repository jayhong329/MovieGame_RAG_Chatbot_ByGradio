from dotenv import load_dotenv
import os
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from fuzzywuzzy import process
from rank_bm25 import BM25Okapi
from collections import defaultdict
import jieba
from openai import OpenAI

load_dotenv()

# 取得當前工作目錄 (cwd)
cwd = os.getcwd()

# 設定 OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

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
VECTOR_INDEX_PATH = os.path.join(cwd, "vector_data/tokenized_movies_vector.index")
IDS_PATH = os.path.join(cwd, "vector_data/tokenized_movies_ids.pkl")


# 系統提示詞
SYSTEM_PROMPT = """你是一個專業的電影與遊戲推薦助理，專門回答關於「電影、遊戲、卡通、動畫、漫畫」的問題。
**請避免回答政治、宗教、新聞、體育、股票、教育等主題**。
如果用戶的問題超出這些範圍，請回覆：
『抱歉，我僅限於電影、遊戲相關話題喔！您可以嘗試在網路上搜尋相關資訊。』

--- 檢索結果 ---
{retrieved_texts}

--- 回答規則 ---
1️ **優先使用檢索資訊** 回答問題
2️ **若檢索不到相關內容**，請先明確說明：「在資料庫中未找到相關資訊」，然後基於你的知識回答
3️ **回答需簡潔但包含重要細節**
4️ **回答請使用正體中文**
5️ **確保資訊準確**，必要時提供額外的建議
"""


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
    """載入事先計算好的FAISS索引和電影.遊戲ID映射"""
    try:
        # 載入電影與遊戲FAISS索引
        print("載入電影與遊戲FAISS索引...")
        movie_index = faiss.read_index(VECTOR_INDEX_PATH)
        game_index = faiss.read_index(GAMES_INDEX_PATH)
            
        # 載入電影與遊戲 ID 映射
        print("載入電影與遊戲 ID 映射...")
        with open(IDS_PATH, 'rb') as f:
            movie_ids = pickle.load(f)
        with open(GAMES_IDS_PATH, 'rb') as f:
            game_ids = pickle.load(f)
            
        return movie_index, movie_ids, game_index, game_ids
    except Exception as e:
        print(f"載入索引時發生錯誤：{str(e)}")
        raise e
    
movie_index, movie_ids, game_index, game_ids = load_index_and_mappings()
    
def search_by_title(user_query, is_movie=True, top_k=5):
    """使用 FuzzyWuzzy 進行名稱查詢"""
    # 根據 is_movie 來選擇查詢的資料集
    if is_movie:
        best_match = process.extractOne(user_query, movies_data["movie_title"])
        if best_match and best_match[1] >= 80:  # 設定最低匹配分數，例如 80
            return best_match[0], best_match[1]
        else:
            return None, 0  # 若匹配度過低則返回 None
    else:
        best_match = process.extractOne(user_query, games_data["game_title"])

    print(best_match)
    return best_match[0], best_match[1]  # 返回最匹配的名稱和分數    

def search_by_genre(user_query, is_movie=True, top_k=5):
    """使用 BM25 + One-Hot Encoding 進行類型查詢"""
    try:
        # 修正：使用正確的資料集和欄位名稱
        if is_movie:
            genres = movies_data["movie_genre"].dropna().tolist()
        else:
            # 這裡原本錯誤地使用了 movies_data
            genres = games_data["game_genre"].dropna().tolist()
            
        tokenized_genres = [str(g).lower().split() for g in genres]
        bm25 = BM25Okapi(tokenized_genres)
        scores = bm25.get_scores(user_query.lower().split())
        
        # 確保索引在有效範圍內
        valid_scores = [(i, s) for i, s in enumerate(scores) if i < len(genres)]
        top_indices = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in top_indices:
            if is_movie and idx < len(movies_data):
                results.append((movies_data.iloc[idx]['movie_title'], score))
            elif not is_movie and idx < len(games_data):
                results.append((games_data.iloc[idx]['game_title'], score))
                
        return results
    except Exception as e:
        print(f"類型搜尋時發生錯誤：{str(e)}")
        return []

def search_by_semantic(user_query, is_movie=True, top_k=5):
    """使用 FAISS 索引 + 餘弦相似度搜尋"""
    try:
        query_vector = model.encode([user_query], normalize_embeddings=True).astype('float32')
        index = movie_index if is_movie else game_index
        
        # 使用 FAISS 搜尋前 top_k 筆
        D, I = index.search(query_vector, top_k)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            # 確保索引在有效範圍內
            if is_movie and 0 <= idx < len(movies_data):
                title = movies_data.iloc[idx]['movie_title']
                results.append((title, float(score)))
            elif not is_movie and 0 <= idx < len(games_data):
                title = games_data.iloc[idx]['game_title']
                results.append((title, float(score)))
                
        return results
    except Exception as e:
        print(f"語義搜尋時發生錯誤：{str(e)}")
        return []

def preprocess_query(user_query):
    """使用結巴斷詞處理查詢字串"""
    try:
        # 使用結巴斷詞
        words = jieba.cut(user_query, cut_all=False)
        # 過濾停用詞（可以自定義停用詞列表）
        stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '影'}
        filtered_words = [word for word in words if word not in stop_words]
        # 重新組合成字串
        processed_query = ' '.join(filtered_words)
        print(f"處理後的查詢: {processed_query}")

        return processed_query
    except Exception as e:
        print(f"查詢預處理時發生錯誤：{str(e)}")
        return user_query

def weighted_search(user_query, is_movie=True, weights=(0.5, 0.2, 0.3), top_k=5):
    """整合多種搜尋方式的加權搜尋"""
    try:
        # 預處理查詢字串
        processed_query = preprocess_query(user_query)
        
        # 獲取各種搜尋結果
        title_match, title_score = search_by_title(user_query, is_movie=is_movie)
        genre_results = search_by_genre(processed_query, is_movie=is_movie, top_k=top_k)
        semantic_results = search_by_semantic(processed_query, is_movie=is_movie, top_k=top_k)
        
        # 初始化組合分數字典
        combined_scores = defaultdict(float)
        
        # 加入標題匹配分數
        combined_scores[title_match] += title_score * weights[0]
        
        # 加入類型匹配分數
        for title, score in genre_results:
            combined_scores[title] += score * weights[1]
            
        # 加入語義相似度分數
        for title, score in semantic_results:
            combined_scores[title] += score * weights[2]
        
        # 排序並返回前 top_k 個結果
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_results
    except Exception as e:
        print(f"加權搜尋時發生錯誤：{str(e)}")
        return []

def get_item_details(item_name, is_movie=True):
    """獲取項目詳細信息"""
    try:
        if is_movie:
            matches = movies_data[movies_data['movie_title'] == item_name]
            if matches.empty:
                print(f"找不到電影：{item_name}")
                return None
            item = matches.iloc[0]
            return {
                'type': '電影',
                'title': item['movie_title'],
                'genre': item.get('movie_genre', '未知類型'),
                'release_date': item.get('release_date', '未知日期'),
                'description': item.get('movie_description', '無簡介'),
                'rating': item.get('vote_average', '無評分')
            }
        else:
            matches = games_data[games_data['game_title'] == item_name]
            if matches.empty:
                print(f"找不到遊戲：{item_name}")
                return None
            item = matches.iloc[0]
            return {
                'type': '遊戲',
                'title': item['game_title'],
                'genre': item.get('game_genre', '未知類型'),
                'release_date': item.get('release_date', '未知日期'),
                'description': item.get('game_description', '無簡介'),
                'production': item.get('game_production', '未知製作商')
            }
    except Exception as e:
        print(f"獲取項目詳情時發生錯誤：{str(e)}")
        return None

def get_ai_response(query, retrieved_texts):
    """使用 OpenAI API 生成流式回應"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "system", "content": f"以下是參考資訊：\n{retrieved_texts}"}
        ]

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
        # 判斷查詢類型
        is_movie_query = "電影" in message or "movie" in message.lower()
        is_game_query = "遊戲" in message or "game" in message.lower()
        
        # 獲取搜尋結果
        similar_items_info = []
        
        if is_movie_query or not is_game_query:
            movie_results = weighted_search(message, is_movie=True)
            for title, score in movie_results:
                details = get_item_details(title, is_movie=True)
                if details:
                    similar_items_info.append(
                        f"類型：{details['type']}\n"
                        f"名稱：{details['title']}\n"
                        f"類型：{details['genre']}\n"
                        f"發行日：{details['release_date']}\n"
                        f"簡介：{details['description']}\n"
                        f"評分：{details['rating']}\n"
                        f"相似度分數：{score:.4f}\n"
                    )
                    
        if is_game_query or not is_movie_query:
            game_results = weighted_search(message, is_movie=False)
            for title, score in game_results:
                details = get_item_details(title, is_movie=False)
                if details:
                    similar_items_info.append(
                        f"類型：{details['type']}\n"
                        f"名稱：{details['title']}\n"
                        f"類型：{details['genre']}\n"
                        f"發行日：{details['release_date']}\n"
                        f"簡介：{details['description']}\n"
                        f"遊戲製造商：{details['production']}\n"
                        f"相似度分數：{score:.4f}\n"
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