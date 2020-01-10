"""Generates the pre-processed text corpus from a MySQL database."""
import allnews_am.db
import allnews_am.processing


mysql_db = allnews_am.db.MySQL()
# TODO(vt257) Add code that performs processing and writes to a text file.
news = mysql_db.fetch_news(1)
print([allnews_am.processing.tokenize(n) for n in news])
print(news)
