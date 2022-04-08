import sqlite3

create_db_cmd = """
CREATE TABLE IF NOT EXISTS TWITTER_SENTIMENT
(TWEETS         TEXT    NOT NULL,
PREDICTION             TEXT    NOT NULL,
FEEDBACK               TEXT    NOT NULL);
"""

query_for_all = """
SELECT * FROM TWITTER_SENTIMENT
"""

insert_query = """
INSERT INTO TWITTER_SENTIMENT VALUES (?, ?, ?);
"""

db_name = "twitter_sentiment_analysis.db"


def opendb(db_name):
    """
    Function to connect to the DB.
    """
    try:
        conn = sqlite3.connect(db_name)
    except sqlite3.Error:
        return False
    cur = conn.cursor()
    return [conn, cur]


def insert_tweet(tweet, prediction, feedback):
    """"
    Function to insert data into DB
    """
    conn, cur = opendb(db_name)
    res = cur.execute(insert_query, (tweet, prediction, feedback))
    conn.commit()
    conn.close
    return "Inserted Successfully!"    


def create_db():
    """
    Function to create DB
    """
    conn, cur = opendb(db_name)
    cur.execute(create_db_cmd)
    conn.close
    return "DB Created Successfully!"


def query_all():
    """
    Function to query all the data from DB
    """
    conn, cur = opendb(db_name)
    res = cur.execute(query_for_all)
    conn.close
    return res
