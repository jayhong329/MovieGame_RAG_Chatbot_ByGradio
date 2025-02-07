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
MODEL_NAME = 'sentence-transformers/multi-qa-distilbert-cos-v1'
# MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
# MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v1'

MOVIES_EXCEL = './tmdb_top_rated_movies_total.xlsx'
GAMES_EXCEL = './game_data(1980~2017).xlsx'
MOVIES_INDEX_PATH = './movies_excel_vector.index'
GAMES_INDEX_PATH = './games_excel_vector.index'
MOVIES_IDS_PATH = './movies_excel_ids.pkl'
GAMES_IDS_PATH = './games_excel_ids.pkl'


def read_excel_data(file_path, is_movie=True):
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


def create_vector_index_from_excel(data, is_movie=True):
    """從Excel創建向量索引"""
    try:
        # 載入模型
        print(f"載入 {MODEL_NAME} 模型中...")
        model = SentenceTransformer(MODEL_NAME)

        # 根據數據類型生成嵌入向量
        if is_movie:
            combined_texts = [
                f"{row['movie_title']} - {row['movie_description']} - {row['movie_genre']}"
                for _, row in data.iterrows()
            ]
            index_path = MOVIES_INDEX_PATH
            ids_path = MOVIES_IDS_PATH
        else:
            combined_texts = [
                f"{row['game_title']} - {row['game_description']} - {row['game_genre']}"
                for _, row in data.iterrows()
            ]
            index_path = GAMES_INDEX_PATH
            ids_path = GAMES_IDS_PATH

        print("生成嵌入向量...")
        embeddings = model.encode(
            combined_texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # 創建 FAISS 索引
        print("創建 FAISS 索引...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)

        # 添加向量到索引
        ids = data['id'].values.astype('int64')
        index.add_with_ids(embeddings.astype('float32'), ids)

        # 保存索引
        print(f"保存索引到 {index_path}")
        faiss.write_index(index, index_path)

        # 保存 ID 映射
        print("保存 ID 映射...")
        if is_movie:
            ids_dict = {row['id']: row['movie_title'] for _, row in data.iterrows()}
        else:
            ids_dict = {row['id']: row['game_title'] for _, row in data.iterrows()}
            
        with open(ids_path, 'wb') as f:
            pickle.dump(ids_dict, f)

        print("向量索引創建成功")
    except Exception as e:
        print(f"創建向量索引時發生錯誤：{str(e)}")
        raise e

def main():
    try:
        # 分別讀取電影和遊戲數據
        movies_data = read_excel_data(MOVIES_EXCEL, is_movie=True)
        games_data = read_excel_data(GAMES_EXCEL, is_movie=False)

        # 分別創建向量索引
        create_vector_index_from_excel(movies_data, is_movie=True)
        create_vector_index_from_excel(games_data, is_movie=False)
        
        print("所有操作完成！")
    except Exception as e:
        print(f"程序執行時發生錯誤：{str(e)}")
        raise e

if __name__ == "__main__":
    main()