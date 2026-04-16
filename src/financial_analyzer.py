import json
import ollama
import re

class FinancialAnalyzer:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name

    def generate_roi_scenario(self, disease_name: str, area_sqm: int) -> dict:
        system_prompt = "농업 재무 전문가로서 시나리오를 JSON으로 작성하십시오."
        user_prompt = f"질병: {disease_name}, 면적: {area_sqm}m2"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': f"{system_prompt}\n{user_prompt}"}],
                options={'temperature': 0.1}
            )
            res_text = response['message']['content']
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            clean_text = json_match.group(0) if json_match else res_text
            return json.loads(clean_text)
        except Exception:
            return {"no_action_loss": "계산 중", "early_action_cost": "계산 중", "net_benefit": "0", "roi_percentage": "0", "scenario_summary": "재무 모델 로딩 중입니다."}
