import os
import json
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

    def generate_prescription(self, disease_name: str) -> dict:
        """
        벡터 검색 후 ollama.chat()을 직접 호출하여 처방 JSON 반환
        """
        # 1. DB 검색 (메타데이터 필터링 적용)
        docs = self.vectorstore.similarity_search(
            query="추천 화학적 방제법과 재배적 방제법을 알려줘",
            k=3,
            filter={"disease_name": disease_name}
        )

        # 2. 검색된 문서 텍스트 결합
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # 3. 프롬프트 구성
        system_prompt = """당신은 농작물 병해충 방제 전문가 'Agri-Doctor'입니다.
제공된 [방제 매뉴얼 문맥]만을 기반으로 처방전을 작성해야 합니다.
없는 농약이나 임의의 수치를 지어내지 마십시오.

반드시 아래의 JSON 형식으로만 응답하십시오. 다른 텍스트는 추가하지 마십시오.
{
    "disease_name": "진단된 질병명을 그대로 기입하세요",
    "environment_and_symptoms": "매뉴얼에서 발생 환경과 증상을 찾아 1~2줄로 요약해서 작성하세요",
    "cultural_control": "매뉴얼에서 농약을 제외한 재배적 방제법을 찾아 1~2줄로 요약해서 작성하세요",
    "chemical_control": [
        {"pesticide_name": "농약명", "dilution": "희석배수", "usage": "사용기준"}
    ],
    "llm_narrative": "위 정보를 바탕으로 농민을 위한 전문가 조언을 3문장 이내로 작성하세요"
}"""

        user_prompt = f"""[방제 매뉴얼 문맥]
{context_text}

[진단된 질병명]
{disease_name}

위 정보를 바탕으로 처방 JSON을 생성하십시오."""

        # 4. Ollama 직접 호출
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'temperature': 0.1}
            )

            response_text = response['message']['content']

            # JSON 파싱 전처리
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)

        except Exception as e:
            return {"error": f"LLM 호출 또는 JSON 파싱 실패: {str(e)}"}

# 실행 블록
if __name__ == "__main__":
    rag_engine = AgriDoctorRAG(llm_model="gemma:2b")

    target_disease = "고추 탄저병"
    print(f"[{target_disease}] 처방 생성 중...\n")

    result_json = rag_engine.generate_prescription(target_disease)
    print(json.dumps(result_json, indent=4, ensure_ascii=False))
