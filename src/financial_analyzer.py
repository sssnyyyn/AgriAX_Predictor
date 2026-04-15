import json
import ollama

class FinancialAnalyzer:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name

    def generate_roi_scenario(self, disease_name: str, area_sqm: int) -> dict:
        system_prompt = """당신은 농업 재무 분석 전문가입니다.
주어진 질병과 농지 면적을 바탕으로 가상의 재무 시나리오를 계산하여 아래 JSON 형식으로만 응답하십시오. 다른 말은 절대 추가하지 마세요.
{
    "no_action_loss": "방치 시 예상 손실액 (단위: 원)",
    "early_action_cost": "초기 방제 비용 (단위: 원)",
    "net_benefit": "방제로 인한 순수익 (단위: 원)",
    "roi_percentage": "투자수익률 (%)",
    "scenario_summary": "농가 경영 의사결정을 위한 2문장 이내의 객관적 조언"
}"""

        user_prompt = f"질병명: {disease_name}\n농지 면적: {area_sqm}제곱미터\n위 조건으로 재무 분석 JSON을 생성하십시오."

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'temperature': 0.1} # 일관성을 위해 온도 낮춤
            )

            # Markdown 기호 제거 및 JSON 파싱
            clean_text = response['message']['content'].replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)

        except Exception as e:
            return {"error": f"Ollama 응답 처리 오류: {str(e)}\n원문: {response['message']['content'] if 'response' in locals() else '응답 없음'}"}
