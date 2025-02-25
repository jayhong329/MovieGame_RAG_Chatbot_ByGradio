import pymysql
import json

# 資料庫連線
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='P@ssw0rd',
    database='tmdb_movie_data',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        # 查詢 genres 表中的所有 id 和 name
        cursor.execute("SELECT id, name FROM genres")
        genre_mapping = {row['id']: row['name'] for row in cursor.fetchall()}  # 建立 id 和 name 的映射

        # 查詢 movie_top_rated 表中的所有電影及其 genre_ids
        cursor.execute("SELECT id, genre_ids FROM movie_top_rated")
        movies = cursor.fetchall()

        # 插入到關聯表 movie_genres
        insert_sql = "INSERT INTO movie_genres (movie_id, genre_name) VALUES (%s, %s)"
        for movie in movies:
            movie_id = movie['id']
            genre_ids = json.loads(movie['genre_ids'])  # genre_ids 是 JSON 格式的字符串，需解析為列表

            for genre_id in genre_ids:
                if genre_id in genre_mapping:  # 確保 genre_id 存在於 genres 表中
                    genre_name = genre_mapping[genre_id]
                    cursor.execute(insert_sql, (movie_id, genre_name))

        # 提交變更
        connection.commit()
        print("成功插入關聯表資料")

except Exception as e:
    connection.rollback()
    print("執行失敗:", e)

finally:
    connection.close()
