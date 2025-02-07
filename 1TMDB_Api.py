import requests as req
import pymysql
from dotenv import load_dotenv
import json
import os

# 加载 .env 文件
load_dotenv()

API_TOKEN = os.getenv("TMDB_APIKey")

# 資料庫連線
connection = pymysql.connect(
    host = 'localhost',
    user = 'root',
    password = 'P@ssw0rd',
    database = 'tmdb_movie_data',
    charset = 'utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

def fetch_tmdb_data(page):
    # 建立 list 來放置列表資訊
    # list_posts = []

    url = f"https://api.themoviedb.org/3/movie/top_rated?language=zh-tw&page={page}"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = req.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"API 請求失敗，狀態碼: {response.status_code}, 錯誤信息: {response.text}")
    return response.json()['results']  # 返回 `results` 列表

# 起始頁數
init_page = 1

# 最新頁數
latest_page = 20

# 在已經知道分頁數的情況下
for page in range(init_page, latest_page + 1):
    print(f"正在處理第 {page} 頁...")
    posts = fetch_tmdb_data(page)  # 獲取當前頁的所有資料

    try:
        with connection.cursor() as cursor:  # 使用 with 語法自動處理游標的打開和關閉
            # 直接寫入資料 (新舊資料都在)
            sql = "INSERT INTO `movie_top_rated` (`title`, `overview`,`genre_ids`, `release_date`, `vote_average`, `poster_path`) VALUES (%s, %s, %s, %s, %s, %s)"

            # 準備要插入的資料
            insert_data = []
            for post in posts:
                try:
                    insert_data.append((
                        post['title'], 
                        post['overview'], 
                        json.dumps(post['genre_ids']), 
                        post['release_date'], 
                        post['vote_average'], 
                        post['poster_path']
                    ))
                except KeyError as e:
                    print(f"跳過資料，因為缺失字段: {e}")
            
            # 使用 executemany() 插入多筆資料
            if insert_data:
                cursor.executemany(sql, insert_data)
                connection.commit()
                print(f"成功插入 {len(insert_data)} 筆資料")
            else:
                print("沒有可插入的資料")

    except Exception as e:
        # 回滾
        connection.rollback()
        print("SQL 執行失敗")
        print(e)

# 關閉資料庫連線
connection.close()