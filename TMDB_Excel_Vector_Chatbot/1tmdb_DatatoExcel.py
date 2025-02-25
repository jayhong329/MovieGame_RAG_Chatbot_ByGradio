import requests
import pandas as pd
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# API 配置
API_KEY = os.getenv("TMDB_APIKey")  # TMDB API 密鑰

# Excel 文件路徑
EXCEL_PATH = "tmdb_top_rated_movies.xlsx"

def fetch_movies_from_api(max_pages=500):
    """從 TMDB API 抓取電影資訊"""
    movies = []
    page = 301

    # url = f"https://api.themoviedb.org/3/movie/top_rated?language=zh-tw&page={page}"

    # headers = {
    #     "accept": "application/json",
    #     "Authorization": f"Bearer {API_KEY}"
    # }

    while page <= max_pages:
        try:
            # 动态构建 URL
            url = f"https://api.themoviedb.org/3/movie/top_rated"
            
            # 参数和头部
            params = {
                "language": "zh-tw",
                "page": page,
            }
            headers = {"Authorization": f"Bearer {API_KEY}"}

            # 發送 GET 請求
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # 如果返回狀態碼不為 200，拋出異常

            data = response.json()
            results = data.get("results", [])

            # 如果結果為空，停止抓取
            if not results:
                break

            # 提取需要的字段
            for movie in results:
                movies.append({
                    "id": movie.get("id"),
                    "title": movie.get("title"),
                    "overview": movie.get("overview"),
                    "release_date": movie.get("release_date"),
                    "vote_average": movie.get("vote_average"),
                    "genre_ids": movie.get("genre_ids"),
                    "poster_path": movie.get("poster_path"),
                })

            # 打印進度
            print(f"成功抓取第 {page} 頁數據，共 {len(results)} 條紀錄。")
            page += 1

            # 停止條件（可自定義，例如限制頁數）
            if page > 500:  # 抓取前 50 頁
                break

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP 錯誤：{http_err}")
            break

        except Exception as e:
            print(f"抓取第 {page} 頁時發生錯誤：{str(e)}")
            break

    return movies

def save_to_excel(movies, excel_path):
    """將電影資訊保存到 Excel 文件"""
    try:
        # 將數據轉換為 DataFrame
        df = pd.DataFrame(movies)

        # 保存為 Excel 文件
        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"電影資訊已成功保存到 {excel_path}")
    except Exception as e:
        print(f"保存到 Excel 時發生錯誤：{str(e)}")

def main():
    # 抓取電影資訊
    movies = fetch_movies_from_api(max_pages=500)
    
    if movies:
        # 保存到 Excel
        save_to_excel(movies, EXCEL_PATH)
    else:
        print("沒有抓取到任何電影資訊。")

if __name__ == "__main__":
    main()