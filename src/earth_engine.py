import ee
import time
import pandas as pd
import numpy as np
import streamlit as st

class EarthEngineManager:
    """Google Earth Engine 통신 및 위성 데이터 전처리 전담 클래스"""

    @staticmethod
    @st.cache_resource
    def initialize(project_id=""):
        try:
            if "GCP_SERVICE_ACCOUNT" in st.secrets:
                info = dict(st.secrets["GCP_SERVICE_ACCOUNT"])
                private_key = info['private_key'].replace('\\n', '\n')
                credentials = ee.ServiceAccountCredentials(
                    info['client_email'],
                    key_data=private_key
                )
                ee.Initialize(credentials=credentials, project=project_id)
                return True, "SUCCESS"
            else:
                if project_id:
                    ee.Initialize(project=project_id)
                else:
                    ee.Initialize()
                return True, "SUCCESS"
        except Exception as e:
            return False, str(e)

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_real_gee_ndvi(lon, lat):
        """특정 좌표(lon, lat)의 최근 40일치 Sentinel-2 이미지를 조회하여 14일 분량의 NDVI 추출"""
        try:
            poi = ee.Geometry.Point([lon, lat])
            end_date = ee.Date(int(time.time() * 1000))
            start_date = end_date.advance(-40, 'day')

            s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                   .filterBounds(poi) \
                   .filterDate(start_date, end_date) \
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

            def get_ndvi(image):
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return image.addBands(ndvi)

            s2_ndvi = s2.map(get_ndvi)
            ndvi_list = s2_ndvi.select('NDVI').getRegion(poi, 10).getInfo()

            if len(ndvi_list) <= 1:
                return None, "위성 데이터가 존재하지 않거나 구름이 너무 많습니다"

            df = pd.DataFrame(ndvi_list[1:], columns=ndvi_list[0])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.groupby('time').mean(numeric_only=True).reset_index()
            df = df.sort_values('time')

            ndvi_values = df['NDVI'].dropna().values

            if len(ndvi_values) == 0:
                return None, "유효한 NDVI 픽셀을 찾을 수 없습니다"

            # 14일 데이터 보정
            if len(ndvi_values) < 14:
                pad_length = 14 - len(ndvi_values)
                padded = np.pad(ndvi_values, (pad_length, 0), mode='edge')
                return padded, "SUCCESS"
            else:
                return ndvi_values[-14:], "SUCCESS"

        except Exception as e:
            return None, str(e)
