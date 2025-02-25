import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 全局設置
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EXCEL_PATH = "../tmdb_top_rated_movies_total.xlsx"
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
        
        # 去除空值
        data['movie_description'] = data['movie_description'].dropna()

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
            str(row['movie_description']).strip()  # 確保是字符串
            for _, row in data.iterrows()
        ]
        
        print("檢查數據樣本：", combined_texts[:3])  # 查看前三筆數據是否正常
        
        print("生成嵌入向量...")
        embeddings = model.encode(
            combined_texts,
            batch_size=16,
            show_progress_bar=True,
            # normalize_embeddings=True
        )

        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"嵌入向量格式錯誤，應該是 np.ndarray，但獲得 {type(embeddings)}")
        
        print(f"嵌入向量維度: {embeddings.shape}")

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # 正規化

        # 創建 FAISS 索引
        print("創建 FAISS 索引...")
        dimension = embeddings.shape[1]
        if not isinstance(dimension, int):
            raise ValueError(f"維度錯誤，期望 int，但獲得 {type(dimension)}: {dimension}")

        print("創建 FAISS 索引...")
        index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(index)

        # **使用 DataFrame 行索引作為 FAISS ID**
        tmdb_ids = data['id'].values.astype('int64') # 使用 DataFrame 行索引作為 FAISS ID
        index.add_with_ids(embeddings.astype('float32'), tmdb_ids)

        # 保存索引
        print(f"保存索引到 {INDEX_PATH}")
        faiss.write_index(index, INDEX_PATH)

        # **修正 ID 映射 (FAISS ID → TMDB ID)**
        print("保存電影 ID 映射...")
        tmdb_ids_dict = {i: row['id'] for i, row in data.iterrows()}
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