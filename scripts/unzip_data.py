import zipfile
import os
import time

# 네 컴퓨터의 실제 절대 경로
zip_path = r"C:\Users\82108\Documents\GitHub\AgriAX_Predictor\data\raw\archive.zip"
extract_path = r"C:\Users\82108\Documents\GitHub\AgriAX_Predictor\data\raw\plantvillage"

os.makedirs(extract_path, exist_ok=True)

print("🚀 파이썬으로 초고속 압축 해제를 시작합니다. 잠시만 기다려주세요...")
start_time = time.time()

# 압축 해제 실행
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

end_time = time.time()
print(f"✅ 압축 해제 완료! 걸린 시간: {end_time - start_time:.2f}초")
