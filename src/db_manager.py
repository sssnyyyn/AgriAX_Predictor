import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'agriax_history.db')

class DatabaseManager:
    @staticmethod
    def init_db():
        # data 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS daily_monitoring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_date TEXT,
                disease_name TEXT,
                loss_amount REAL,
                cost_amount REAL,
                net_profit REAL
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def insert_record(disease, loss, cost, profit):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''
            INSERT INTO daily_monitoring (record_date, disease_name, loss_amount, cost_amount, net_profit)
            VALUES (?, ?, ?, ?, ?)
        ''', (date_str, disease, loss, cost, profit))
        conn.commit()
        conn.close()

    @staticmethod
    def get_history():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM daily_monitoring ORDER BY record_date ASC", conn)
        conn.close()
        return df

    @staticmethod
    def insert_mock_data():
        """
        시계열 데이터 시각화를 위해 초기 DB가 비어있을 경우 5일치 샘플 데이터를 자동 삽입합니다.
        """
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM daily_monitoring")

        if c.fetchone()[0] == 0:
            base_date = datetime.now() - timedelta(days=5)
            mock_data = [
                ((base_date + timedelta(days=0)).strftime("%Y-%m-%d 10:00:00"), '정상', 0, 0, 0),
                ((base_date + timedelta(days=1)).strftime("%Y-%m-%d 14:30:00"), '고추 탄저병', 2500000, 495000, 1505000),
                ((base_date + timedelta(days=2)).strftime("%Y-%m-%d 09:15:00"), '고추 탄저병', 3200000, 495000, 2065000),
                ((base_date + timedelta(days=3)).strftime("%Y-%m-%d 16:45:00"), '고추 탄저병', 4100000, 495000, 2785000),
                ((base_date + timedelta(days=4)).strftime("%Y-%m-%d 11:20:00"), '정상 (방제완료)', 0, 0, 0)
            ]
            c.executemany('''
                INSERT INTO daily_monitoring (record_date, disease_name, loss_amount, cost_amount, net_profit)
                VALUES (?, ?, ?, ?, ?)
            ''', mock_data)
            conn.commit()

        conn.close()
