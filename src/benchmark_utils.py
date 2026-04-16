import psutil
import os
import time
import pandas as pd
from datetime import datetime

class ModelBenchmark:
    @staticmethod
    def get_memory_usage():
        """현재 프로세스의 메모리 소모량을 MB 단위로 반환합니다."""
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        return round(mem_bytes / (1024 * 1024), 2)

    @staticmethod
    def log_to_csv(metrics_dict, file_name="data/model_comparison.csv"):
        """데이터를 CSV 파일에 누적하여 저장합니다."""
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # 기록 시간 추가
        metrics_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([metrics_dict])

        if not os.path.isfile(file_name):
            df.to_csv(file_name, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(file_name, mode='a', header=False, index=False, encoding='utf-8-sig')

def run_comparison_test(model_name, test_func, *args, **kwargs):
    """모델별 벤치마크를 수행하는 메인 실행 함수입니다."""
    mem_before = ModelBenchmark.get_memory_usage()
    start_time = time.time()

    # 모델 실행 (추론)
    result = test_func(*args, **kwargs)

    end_time = time.time()
    mem_after = ModelBenchmark.get_memory_usage()

    metrics = {
        "model_name": model_name,
        "latency_sec": round(end_time - start_time, 2),
        "ram_usage_mb": round(mem_after - mem_before, 2),
        "peak_ram_mb": mem_after,
        "status": "success" if result else "fail"
    }

    ModelBenchmark.log_to_csv(metrics)
    return metrics
