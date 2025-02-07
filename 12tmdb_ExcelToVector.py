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

EXCEL_PATH = './tmdb_top_rated_movies_total.xlsx'
INDEX_PATH = './tmdb_excel_vector.index'
TMDB_IDS_PATH = './tmdb_excel_ids.pkl'


def read_excel_data(file_path):
    """從 Excel 文件讀取電影數據"""
    try:
        print(f"從 {file_path} 讀取電影數據...")
        data = pd.read_excel(file_path)

        # 检查是否有数据
        if data.empty:
            raise ValueError(f"Excel 文件 {file_path} 中沒有數據。")
        
        # 填补空值
        data['overview'] = data['overview'].fillna('')

        print("Excel 數據讀取完成。")
        return data
    except Exception as e:
        print(f"讀取 Excel 文件時發生錯誤：{str(e)}")
        raise e


def create_vector_index_from_excel(data):
    """從Excel創建向量索引"""
    try:
        # 載入模型
        print(f"載入  {MODEL_NAME} 模型中...")
        model = SentenceTransformer(MODEL_NAME)

        # 生成嵌入向量
        combined_texts = [
            f"{row['title']} - {row['overview']} - {row['genre_ids']}"
            for _, row in data.iterrows()
        ]
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
        # 从 Excel 文件读取数据
        data = read_excel_data(EXCEL_PATH)

        # 創建向量索引
        create_vector_index_from_excel(data)
        print("所有操作完成！")
    except Exception as e:
        print(f"程序執行時發生錯誤：{str(e)}")
        raise e

if __name__ == "__main__":
    main()