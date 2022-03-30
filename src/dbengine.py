import sqlite3
from pathlib import Path
from typing import Union

class DBEngine:
    """SQLite3 DB Engine Wrapper"""
    def __init__(self, db_path: Union[str, Path]):
        self.conn = sqlite3.connect(db_path)

    def query(self, sql_statement: str):
        cur = self.conn.cursor()
        res = cur.execute(sql_statement)
        res = res.fetchall()
        cur.close()
        return res

    def quit(self):
        self.conn.close()

    def _sql(self, query:str):
        # no auto commit
        cur = self.conn.cursor()
        cur.execute(query)
        self.conn.commit()
        cur.close()

    def insert(self, query:str):
        return self._sql(query)

    def update(self, query:str):
        return self._sql(query)