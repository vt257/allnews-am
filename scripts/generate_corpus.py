"""Generates the pre-processed text corpus from a MySQL database."""
import allnews_am.db

mysql_db = allnews_am.db.MySQL()
# TODO(vt257) Add code that performs processing and writes to a text file.
print(mysql_db.fetch_news(10))
