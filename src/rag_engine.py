import os
import json
import ollama
import time
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.benchmark_utils import run_comparison_test

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

class AgriDoctorRAG:
    def __init__(self, db_path=DEFAULT_DB_PATH, llm_model="gemma:2b"):
        self.llm_model = llm_model
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"벡터 DB 경로를 찾을 수 없습니다: {db_path}")

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def generate_prescription(self, disease_name: str, weather_info: dict = None) -> dict:
        # 1. 문서 검색
        docs = self.vectorstore.similarity_search(
            query="방제법을 알려줘",
            k=3,
            filter={"disease_name": disease_name}
        )
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # 2. 기상 정보 구성
        weather_context = "기상 정보 없음"
        if weather_info:
            weather_context = f"온도 {weather_info.get('temperature')}°C, 습도 {weather_info.get('humidity')}%"

        # 3. 프롬프트 구성
        system_prompt = """당신은 농작물 병해 전문가입니다. 반드시 JSON으로만 답변하세요.
        {
            "disease_name": "질병명",
            "environment_and_symptoms": "발생 원인 및 증상 상세 설명",
            "cultural_control": "재배적 방제법(농약 제외)",
            "chemical_control": [{"pesticide_name": "농약명", "dilution": "배수", "usage": "기준"}],
            "llm_narrative": "기상 상황을 반영한 전문가의 핵심 조언"
        }"""
        user_prompt = f"질병: {disease_name}\n정보: {context_text}\n기상: {weather_context}"

        # 4. 추론 함수 정의
        def run_inference():
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'temperature': 0.1}
            )
            res_text = response['message']['content']
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            return json.loads(json_match.group(0)) if json_match else json.loads(res_text)

        # 5. 실행 및 벤치마크 (중복 실행 방지)
        try:
            result = run_inference()
            run_comparison_test(model_name=self.llm_model, test_func=lambda: result)
            if weather_info:
                result['weather'] = weather_info
            return result
        except Exception as e:
            return {"disease_name": disease_name, "environment_and_symptoms": f"오류 발생: {e}", "cultural_control": "점검 필요", "chemical_control": [], "llm_narrative": "시스템 재시작이 필요합니다."}
