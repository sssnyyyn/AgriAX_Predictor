class DiseaseDictionary:
    """
    작물 병해별 메타데이터, 기준 손실률, 전문가 방제 가이드를 관리하는 클래스
    """

    _MAPPING = {
        0: {"name": "정상 (모든 작물)", "status": "안전", "base_loss": 0.0, "urgency": "불필요", "guide": "1. 특이사항 없음\n2. 현재 상태 유지"},
        1: {"name": "고추 탄저병", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 병든 과실 및 잎 조기 제거\n2. 비 오기 전후 등록 약제 살포"},
        2: {"name": "고추 흰가루병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 초기 병반 발견 시 약제 살포\n2. 밀식 방지 및 통풍 개선"},
        3: {"name": "무 검은무늬병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 병든 잎 제거\n2. 종자 소독 및 윤작 권장"},
        4: {"name": "무 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 다습 환경 개선 (배수 철저)\n2. 적용 보호살균제 살포"},
        5: {"name": "배추 검은썩음병", "status": "심각", "base_loss": 0.20, "urgency": "매우 높음", "guide": "1. 발병 개체 즉시 소각/매몰\n2. 농기구 소독 철저"},
        6: {"name": "배추 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 밀식 피하고 환기 유의\n2. 발병 초기 약제 살포"},
        7: {"name": "애호박 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 잎에 물방울이 맺히지 않도록 관리\n2. 이병엽 제거 및 약제 방제"},
        8: {"name": "애호박 흰가루병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 질소질 비료 과용 금지\n2. 초기 예방 약제 살포"},
        9: {"name": "양배추 균핵병", "status": "심각", "base_loss": 0.15, "urgency": "높음", "guide": "1. 병든 식물체와 흙 제거\n2. 적용 약제 살포 및 벼과 작물 윤작"},
        10: {"name": "양배추 무름병", "status": "심각", "base_loss": 0.20, "urgency": "매우 높음", "guide": "1. 상처로 감염되므로 해충 방제 병행\n2. 이병주 조기 제거"},
        11: {"name": "오이 노균병", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 야간 다습 환경 개선\n2. 예방 위주의 약제 살포"},
        12: {"name": "오이 흰가루병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 건조하지 않도록 관리\n2. 발생 초기부터 약제 교차 살포"},
        13: {"name": "콩 불마름병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 무병 종자 사용\n2. 비 오기 전 예방 약제 살포"},
        14: {"name": "콩 점무늬병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 수확 후 잔재물 제거\n2. 밀식 방지"},
        15: {"name": "토마토 잎마름병", "status": "위험", "base_loss": 0.15, "urgency": "높음", "guide": "1. 하엽 위주로 발병하므로 적엽 실시\n2. 비료 부족 방지 및 약제 살포"},
        16: {"name": "파 검은무늬병", "status": "경고", "base_loss": 0.10, "urgency": "보통", "guide": "1. 병든 잎 조기 제거\n2. 등록 약제 살포"},
        17: {"name": "파 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 배수 관리 철저\n2. 발병 초기 7-10일 간격 약제 살포"},
        18: {"name": "파 녹병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 비료가 부족하지 않게 추비\n2. 적용 약제 살포"},
        19: {"name": "호박 노균병", "status": "위험", "base_loss": 0.12, "urgency": "높음", "guide": "1. 다습한 환경 피하기\n2. 병든 잎 제거 및 약제 살포"},
        20: {"name": "호박 흰가루병", "status": "경고", "base_loss": 0.08, "urgency": "보통", "guide": "1. 통풍 및 채광 개선\n2. 예방적 약제 살포"}
    }

    @classmethod
    def get_info(cls, class_idx):
        if class_idx == -1:
            return {
                "name": "판별 불가",
                "status": "분석 보류",
                "base_loss": 0.0,
                "urgency": "전문가 확인",
                "guide": "1. 미학습 데이터이거나 화질이 낮습니다\n2. 재촬영을 권장합니다"
            }

        return cls._MAPPING.get(
            class_idx,
            {"name": "시스템 에러", "status": "에러", "base_loss": 0.0, "urgency": "-", "guide": "-"}
        )
