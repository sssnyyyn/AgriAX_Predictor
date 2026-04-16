import requests
import random
from datetime import datetime

class WeatherManager:
    @staticmethod
    def get_current_weather(lat=36.4500, lon=126.8000, api_key=None):
        """
        API 키가 주어지면 OpenWeatherMap 실시간 데이터를,
        없으면 현재 시각 기반의 데모용 환경 데이터를 반환합니다.
        """
        if api_key:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
                response = requests.get(url, timeout=3)
                data = response.json()
                return {
                    "temperature": data['main']['temp'],
                    "humidity": data['main']['humidity'],
                    "weather_desc": data['weather'][0]['description'],
                    "is_real": True
                }
            except Exception:
                pass # API 호출 실패 시 아래 데모 데이터로 폴백

        # [데모 모드] 고추 탄저병 발병 시나리오에 맞춘 가상 날씨 생성
        return {
            "temperature": round(random.uniform(25.0, 32.0), 1), # 탄저병 호발 온도
            "humidity": random.randint(75, 95),                 # 높은 습도
            "weather_desc": random.choice(["흐리고 비", "습함", "소나기"]),
            "is_real": False
        }
