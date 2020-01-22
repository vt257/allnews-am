import os
import pymysql.cursors

NEWS_TABLE = 'website_posts'
TEXT_FIELD = 'text'
TITLE_FIELD = 'title'

MYSQL_MAX_LIMIT = 18446744073709551610


class MySQL(object):
    """A MySQL database singleton.

    Attrs:
        connection: The pymysql connection.
    """
    __instance = None

    def __new__(cls, db_host=None, db_name=None, db_user=None, db_pass=None):
        """Initializes the MySQL database singleton.

        Args:
            db_host: The host address. If not supplied, uses either the
                ALLNEWS_AM_MYSQL_HOST environment variable when set, or
                'localhost' otherwise.
            db_name: The name of the database. If not supplied, uses either the
                ALLNEWS_AM_MYSQL_NAME environment variable when set, or
                'news' otherwise.
            db_user: The mysql user. If not supplied, uses the
                ALLNEWS_AM_MYSQL_USER environment variable when set, or 'root'
                otherwise.
            db_pass: The mysql password. If not supplied, uses the
                ALLNEWS_AM_MYSQL_PASS environment variable when set, or 'root'
                otherwise.
        """
        if MySQL.__instance is None:
            MySQL.__instance = object.__new__(cls)
            if db_host is None:
                if 'ALLNEWS_AM_MYSQL_HOST' in os.environ:
                    db_host = os.environ['ALLNEWS_AM_MYSQL_HOST']
                else:
                    db_host = 'localhost'
            if db_name is None:
                if 'ALLNEWS_AM_MYSQL_NAME' in os.environ:
                    db_name = os.environ['ALLNEWS_AM_MYSQL_NAME']
                else:
                    db_name = 'news'
            if db_user is None:
                if 'ALLNEWS_AM_MYSQL_USER' in os.environ:
                    db_user = os.environ['ALLNEWS_AM_MYSQL_USER']
                else:
                    db_user = 'root'
            if db_pass is None:
                if 'ALLNEWS_AM_MYSQL_PASS' in os.environ:
                    db_pass = os.environ['ALLNEWS_AM_MYSQL_PASS']
                else:
                    db_pass = 'root'

            MySQL.__instance.connection = pymysql.connect(
                host=db_host, user=db_user, password=db_pass, db=db_name,
                charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        return MySQL.__instance

    def fetch_news(self, offset=0, limit=MYSQL_MAX_LIMIT):
        """Gets the title and full text field of the news articles.

        Args:
            offset: The offset for the query.
            limit: The number of news articles to fetch. If not set, fetches all
                news articles.

        Returns:
            A sequence of tuples strings corresponding to the title and full
            text of the articles.
        """
        with self.connection.cursor() as cursor:
            sql = (f"SELECT `{TITLE_FIELD}`, `{TEXT_FIELD}` "
                   f"FROM `{NEWS_TABLE}` LIMIT %s, %s")
            cursor.execute(sql, (offset, limit))
            return [
                (news_item[TITLE_FIELD], news_item[TEXT_FIELD])
                for news_item in cursor.fetchall()
            ]
