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
        # 查詢 movie_genres 中的數據並按 movie_id 分組
        cursor.execute("""
            SELECT movie_id, JSON_ARRAYAGG(genre_name) AS genre_names
            FROM movie_genres
            GROUP BY movie_id
        """)
        genre_data = cursor.fetchall()

        # 更新 movie_top_rated 的 genre_ids
        update_sql = "UPDATE movie_top_rated SET genre_ids = %s WHERE id = %s"
        for row in genre_data:
            genre_names = json.dumps(row['genre_names'], ensure_ascii=False)  # 保持中文正常
            cursor.execute(update_sql, (genre_names, row['movie_id']))

        # 提交變更
        connection.commit()
        print("成功更新 movie_top_rated.genre_ids")

except Exception as e:
    connection.rollback()
    print("執行失敗:", e)

finally:
    connection.close()