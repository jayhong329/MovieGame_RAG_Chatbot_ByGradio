import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pymysql
import pickle
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 全局設置
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

INDEX_PATH = './tmdb_vector.index'
TMDB_IDS_PATH = './tmdb_ids.pkl'

# 資料庫連線設置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'P@ssw0rd'),
    'database': os.getenv('DB_NAME', 'tmdb_movie_data'),
    'charset': os.getenv('DB_CHARSET', 'utf8mb4'),
    'cursorclass': pymysql.cursors.DictCursor
}

def fetch_movies_from_db():
    """從資料庫中獲取電影數據"""
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            # 查詢電影數據
            sql = """
            SELECT id, title, overview, genre_ids, vote_average, poster_path
            FROM movie_top_rated
            """
            cursor.execute(sql)
            movies = cursor.fetchall()
        print("成功從資料庫中獲取電影數據")
        return pd.DataFrame(movies)
    except Exception as e:
        print(f"從資料庫獲取電影數據時發生錯誤：{str(e)}")
        raise e
    finally:
        connection.close()

def create_vector_index_from_db():
    """從資料庫創建向量索引"""
    try:
        # 從資料庫讀取數據
        print("從資料庫中讀取數據...")
        data = fetch_movies_from_db()
        
        # 檢查是否有數據
        if data.empty:
            raise ValueError("資料表為空，無法創建向量索引")

        # 填補空值
        data['overview'] = data['overview'].fillna('')
        data['genre_ids'] = data['genre_ids'].fillna('[]')

        # 載入模型
        print("載入模型中...")
        model = SentenceTransformer(MODEL_NAME)

        # 生成嵌入向量
        combined_texts = [
            f"{row['overview']}"
            for _, row in data.iterrows()
        ]
        print("生成嵌入向量...")
        embeddings = model.encode(
            combined_texts,
            batch_size=6,
            show_progress_bar=True,
            # normalize_embeddings=True
        )

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # 正規化

        # 創建 FAISS 索引
        print("創建 FAISS 索引...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(index)

        # 添加向量到索引
        tmdb_ids = data['id'].values.astype('int64')
        index.add_with_ids(embeddings.astype('float32'), tmdb_ids)

        # 保存索引
        print(f"保存索引到 {INDEX_PATH}")
        faiss.write_index(index, INDEX_PATH)

        # 保存電影 ID 映射
        print("保存電影 ID 映射...")
        tmdb_ids_dict = {row['id']: row['title'] for _, row in data.iterrows()}
        with open(TMDB_IDS_PATH, 'wb') as f:
            pickle.dump(tmdb_ids_dict, f)

        print("向量索引創建成功")
    except Exception as e:
        print(f"創建向量索引時發生錯誤：{str(e)}")
        raise e

def main():
    try:
        # 創建向量索引
        create_vector_index_from_db()
        print("所有操作完成！")
    except Exception as e:
        print(f"程序執行時發生錯誤：{str(e)}")
        raise e

if __name__ == "__main__":
    main()