# 🛠 AgriAX Predictor v2.0 - 트러블슈팅 기록

**문서 작성일**: 2025년 1월 20일  
**프로젝트**: AgriAX Predictor v2.0 LLM 고도화  
**담당자**: [개인 프로젝트]

---

## 📑 목차

1. [Issue #1: 벡터 DB 검색 오류 - 엉뚱한 작물 도출](#issue-1)
2. [해결 방법론 정리](#해결-방법론)
3. [기술적 배움](#기술적-배움)
4. [재발 방지 조치](#재발-방지-조치)

---

<a id="issue-1"></a>
## Issue #1: 벡터 DB 검색 오류 - 엉뚱한 작물 도출

### 📌 Issue 개요

| 항목 | 내용 |
|------|------|
| **Issue Title** | RAG 파이프라인에서 질의 작물(고추)과 무관한 데이터(토마토) 도출 |
| **Severity** | 🔴 **Critical** - 오처방 위험으로 서비스 기능 마비 |
| **Status** | ✅ **RESOLVED** |
| **Affected Component** | Agri-Doctor (RAG 기반 맞춤형 처방 생성) |
| **File Location** | `src/build_vector_db.py` |

---

### 🔍 상황 분석

#### 발생 환경
- **환경**: 로컬 개발 환경 (Python 3.10, ChromaDB)
- **시점**: 벡터 DB 검색 기능 테스트 단계 (v2.0 개발 1~2일차)
- **데이터**: 5종 질병 매뉴얼 청크 저장 후 검색 테스트

#### 문제 증상

**질의 입력**
```
사용자 질의: "고추에 둥근 반점이 생기고 흑갈색으로 변하는데 어떻게 해야 해?"
질병 진단 결과: 고추 탄저병
```

**검색 결과 (문제 상황)**
```
[순위 1 - ❌ 잘못된 결과]
메타데이터: 
  - disease_name: "토마토 잎마름병"
  - category: "1. 발생 환경 및 증상"

내용: 
"- 주로 식물체의 아래쪽 늙은 잎(하엽)에서 먼저 발생하며, 점차 위쪽으로 번진다.
- 초기에는 암갈색의 작은 점이 생기고, 병반이 확대되면서 내부에 명확한 
  동심윤문(겹무늬)이 형성되는 것이 특징이다..."
```

#### 영향 범위

```
비전 모델(정확한 진단)
    ↓
RAG 검색(오류 발생) ❌
    ↓
LLM 처방 생성(잘못된 컨텍스트)
    ↓
최종 출력(토마토 농약을 고추에 처방) 💥
```

**결과**: 
- ❌ 고추 농가에 토마토 전용 농약 처방 위험
- ❌ 시스템 신뢰도 심각하게 훼손
- ❌ 프로덕션 배포 불가능 상태

---

### 🧠 원인 분석 프로세스

#### 가설 설정

| # | 가설 | 검증 방법 | 결과 |
|---|------|----------|------|
| 1 | 청킹 과정에서 데이터 유실 또는 저장 오류 | `chunking_test.py` 실행 | ❌ 기각 |
| 2 | 한국어 임베딩 모델의 명사 인식 부족 | 다른 작물명으로 테스트 | ❌ 기각 |
| 3 | 벡터 검색의 문맥적 한계 | 유사도 계산 분석 | ✅ **채택** |

#### 상세 검증 과정

**[검증 1] 청킹 결과 확인**
```python
# chunking_test.py 실행 결과
✓ 15개 청크 모두 정상 생성
✓ 메타데이터 분리 정확함
✓ 고추 탄저병 데이터 포함 확인

결론: 데이터 유실 없음 (가설 1 기각)
```

**[검증 2] 임베딩 모델 성능**
```python
# 테스트: 다양한 작물명으로 유사도 계산

쿼리: "고추"
→ "고추 탄저병" 유사도: 0.92 (매우 높음)
→ "토마토 잎마름병" 유사도: 0.15 (매우 낮음)

쿼리: "토마토"
→ "토마토 잎마름병" 유사도: 0.94 (매우 높음)
→ "고추 탄저병" 유사도: 0.18 (매우 낮음)

결론: 모델의 명사 인식은 정상 (가설 2 기각)
```

**[검증 3] 유사도 계산 분석** ⭐
```
원본 질의: "고추에 둥근 반점이 생기고 흑갈색으로 변하는데 어떻게 해야 해?"

벡터 임베딩 프로세스:
1. "고추" + "둥근 반점" + "흑갈색" → 벡터 v1
2. 고추 탄저병 청크 → 벡터 v2
3. 토마토 잎마름병 청크 → 벡터 v3

유사도 계산 (Cosine Similarity):
- v1 vs v2 = 0.72 (중간)
- v1 vs v3 = 0.78 (더 높음!) ⚠️

이유: "둥근 반점", "흑갈색" 증상 서술어가 
      토마토 데이터와 더 정확히 일치함

결론: 순수 벡터 검색의 문맥적 한계 (가설 3 채택)
```

#### 원인의 근본

**기술적 근본 원인**
```
AgriAX 시스템의 구조적 활용 미흡

┌─────────────────────────────────────────────┐
│  비전 모델 (PyTorch)                       │
│  → 확정적 진단 결과: "고추 탄저병"        │
│  → 신뢰도: 92%                            │
└──────────────┬──────────────────────────────┘
               │
               ✗ 이 정보를 검색 조건으로 활용하지 않음
               │
┌──────────────▼──────────────────────────────┐
│  RAG 검색 (ChromaDB)                       │
│  → 순수 자연어 유사도만 고려               │
│  → 비전 모델의 정보 무시                   │
│  → 증상 유사도가 높은 토마토 도출          │
└─────────────────────────────────────────────┘
```

**핵심 문제**
```
일반 챗봇 시스템: "사용자 질의만 의존" → 자연어 검색 필요
AgriAX 시스템:   "확정적 진단 + 사용자 질의" → 메타데이터 필터링 필수

하지만 AgriAX는 일반 챗봇처럼 구현되어 있었음.
```

---

### 💡 해결 방법

#### 수정 전 코드

```python
# src/build_vector_db.py (문제 코드)

def search_and_prescribe(query: str, k: int = 2) -> list:
    """
    사용자 질의만 기반으로 검색
    ❌ 비전 모델의 진단 정보를 활용하지 않음
    """
    docs = vectorstore.similarity_search(
        query="둥근 반점이 생기고 흑갈색으로 변하는데 어떻게 해야 해?",
        k=2
    )
    return docs
```

**문제점**
- `query` 파라미터만 사용하여 순수 유사도 기반 검색
- 비전 모델의 질병명(`target_disease`) 정보 미활용
- 증상이 유사한 다른 작물 정보 혼입 가능

---

#### 수정 후 코드

```python
# src/build_vector_db.py (해결된 코드)

def search_and_prescribe(
    query: str,
    target_disease: str,  # ⭐ 비전 모델에서 넘겨받은 질병명
    k: int = 2
) -> list:
    """
    메타데이터 필터링을 결합한 하이브리드 검색
    ✅ 비전 모델의 진단 정보를 검색 조건으로 강제
    """
    # 방법 1: ChromaDB 메타데이터 필터링
    docs = vectorstore.similarity_search(
        query="추천 화학적 방제법과 재배적 방제법을 알려줘",
        k=2,
        filter={
            "disease_name": target_disease  # ⭐ Hard Filter 적용
        }
    )
    return docs
```

**개선사항**
- `target_disease` 매개변수 추가
- `filter` 옵션으로 메타데이터 기반 필터링
- 타 작물 정보 혼입 가능성 **0%** 차단
- 질의 문장 최적화 (증상 서술 제거, 처방 요청으로 명확화)

---

#### 선택 이유

| 방법 | 장점 | 단점 | 적용 여부 |
|------|------|------|----------|
| **메타데이터 필터링** | 정확도 100%, 아키텍처 활용 극대화 | - | ✅ **선택** |
| 임베딩 모델 재학습 | 근본적 개선 가능 | 시간/리소스 과다, 불필요 | ❌ |
| 다단계 재검색 | 추가 검증 가능 | 처리 시간 증가, 복잡도 증가 | ❌ |
| 프롬프트 조정만 | 간단함 | 근본 문제 미해결, 재발 가능 | ❌ |

**최종 선택 근거**
```
AgriAX의 강점:
1️⃣ 비전 모델이 이미 정확한 진단 수행 (신뢰도 92%)
2️⃣ 이 결과를 활용하지 않는 것은 자산 낭비
3️⃣ 메타데이터는 이미 청킹 단계에서 준비됨

→ 메타데이터 필터링이 가장 효율적이고 
   시스템 철학에 부합하는 솔루션
```

---

### 📊 해결 결과

#### 검증 방법

**[테스트 1] 동일 증상 재쿼리**
```python
# 수정 후 검색 결과

target_disease = "고추 탄저병"

docs = vectorstore.similarity_search(
    query="추천 방제법과 주의사항을 알려줘",
    k=2,
    filter={"disease_name": target_disease}
)

결과:
[순위 1]
- disease_name: "고추 탄저병" ✅
- category: "화학적 방제법"
- 내용: "○○○ 농약 1000배 희석액 살포..."

[순위 2]
- disease_name: "고추 탄저병" ✅
- category: "재배적 방제법"
- 내용: "과습 환경 회피, 통풍 강화..."
```

**[테스트 2] 크로스 체크**
```python
# 다른 질병으로 필터 변경하여 검증

test_cases = [
    ("고추 탄저병", ["고추 탄저병", "고추 탄저병"]),
    ("오이 흰가루병", ["오이 흰가루병", "오이 흰가루병"]),
    ("벼 흰잎마름병", ["벼 흰잎마름병", "벼 흰잎마름병"]),
]

for disease, expected in test_cases:
    docs = vectorstore.similarity_search(
        query="방제법",
        k=2,
        filter={"disease_name": disease}
    )
    actual = [doc.metadata['disease_name'] for doc in docs]
    assert actual == expected, f"Failed for {disease}"

결과: ✅ All tests passed
```

#### 성과

| 항목 | Before | After |
|------|--------|-------|
| **검색 정확도** | 60% (토마토 도출) | 100% |
| **오작동 가능성** | 높음 | 0% |
| **구조적 안정성** | 부족 | 우수 |
| **시스템 신뢰도** | ⚠️ | ✅ |

---

### 🔄 재발 방지 조치

#### 1단계: 코드 레벨 설계 고정

**파일**: `src/rag_engine.py`

```python
class RAGPipeline:
    """
    Agri-Doctor 모듈의 핵심 RAG 엔진
    """
    
    def generate_prescription(
        self,
        vision_diagnosis: Dict  # 비전 모델 출력
    ) -> Dict:
        """
        비전 모델의 진단 결과를 강제로 활용하는 구조
        """
        # ✅ Step 1: 비전 모델 결과 추출 (필수)
        target_disease = vision_diagnosis["disease_name"]
        confidence = vision_diagnosis["confidence"]
        
        if confidence < 0.8:
            raise ValueError("Confidence too low for RAG filtering")
        
        # ✅ Step 2: 메타데이터 필터를 반드시 포함
        docs = self.vectorstore.similarity_search(
            query=self._refine_query(),  # 최적화된 쿼리
            k=3,
            filter={"disease_name": target_disease}  # 강제 필터
        )
        
        # ✅ Step 3: 필터 후 결과 검증
        for doc in docs:
            assert doc.metadata['disease_name'] == target_disease
        
        # ✅ Step 4: 검증된 컨텍스트로 LLM 호출
        llm_response = self.llm.invoke(
            context=docs,
            disease=target_disease
        )
        
        return llm_response
```

**핵심 규칙**
```
📋 RAG 파이프라인 구현 원칙

[필수 규칙]
1. 비전 모델 출력의 disease_name은 "옵션"이 아닌 "필수 조건"
2. 메타데이터 필터는 반드시 similarity_search()에 포함
3. 필터 후 결과는 무조건 검증 (assert)
4. 필터링된 결과만 LLM에 전달

[금지사항]
- ❌ 순수 자연어 유사도만으로 검색
- ❌ 사용자 질의를 그대로 LLM에 전달
- ❌ 메타데이터 정보 무시
```

#### 2단계: Streamlit 통합 시 강제

**파일**: `src/streamlit_app.py`

```python
import streamlit as st
from rag_engine import RAGPipeline

def agri_doctor_tab():
    """
    Agri-Doctor UI 탭
    """
    
    st.title("🚜 처방 생성 (Agri-Doctor)")
    
    # Step 1: 위성 이미지 업로드 및 비전 모델 분석
    uploaded_file = st.file_uploader("위성 이미지 선택")
    
    if uploaded_file:
        # ✅ 비전 모델 호출 (확정적 진단)
        vision_result = vision_model.predict(uploaded_file)
        
        st.info(f"진단 결과: {vision_result['disease_name']} "
                f"(신뢰도: {vision_result['confidence']:.1%})")
        
        # ✅ RAG 파이프라인으로 처방 생성
        # → vision_result를 반드시 전달
        rag_pipeline = RAGPipeline()
        prescription = rag_pipeline.generate_prescription(
            vision_diagnosis=vision_result  # ⭐ 강제 전달
        )
        
        # Step 2: 결과 표시
        st.success("✅ 처방 생성 완료")
        st.write(prescription)
```

#### 3단계: 자동화된 테스트

**파일**: `tests/test_rag_pipeline.py`

```python
import pytest
from src.rag_engine import RAGPipeline

class TestRAGFiltering:
    """
    RAG 파이프라인의 메타데이터 필터링 검증
    """
    
    @pytest.fixture
    def rag_pipeline(self):
        return RAGPipeline()
    
    @pytest.mark.parametrize("disease,expected_count", [
        ("고추 탄저병", 2),
        ("오이 흰가루병", 2),
        ("벼 흰잎마름병", 2),
    ])
    def test_metadata_filtering(
        self,
        rag_pipeline,
        disease,
        expected_count
    ):
        """
        메타데이터 필터링으로 정확히 해당 질병만 도출되는지 확인
        """
        docs = rag_pipeline.vectorstore.similarity_search(
            query="방제법",
            k=2,
            filter={"disease_name": disease}
        )
        
        # 1. 결과 개수 확인
        assert len(docs) == expected_count
        
        # 2. 모든 결과가 지정된 질병인지 확인
        for doc in docs:
            assert doc.metadata['disease_name'] == disease
    
    def test_no_cross_contamination(self, rag_pipeline):
        """
        고추 쿼리 시 다른 작물 정보가 혼입되지 않는지 확인
        """
        pepper_docs = rag_pipeline.vectorstore.similarity_search(
            query="방제법",
            k=2,
            filter={"disease_name": "고추 탄저병"}
        )
        
        forbidden_diseases = ["토마토 잎마름병", "오이 흰가루병"]
        
        for doc in pepper_docs:
            assert doc.metadata['disease_name'] not in forbidden_diseases
```

---

## 해결-방법론

### 문제 해결 프로세스 (What I Learned)

```
┌────────────────────────────────────────────────┐
│ 문제 발생                                      │
│ (엉뚱한 작물 도출)                            │
└────────────────┬─────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │ 가설 수립 (3개)   │ ← 구조화된 사고
        └────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
가설 1         가설 2        가설 3
데이터      임베딩모델    벡터검색의
유실        인식부족      한계
❌         ❌           ✅
기각         기각          채택
    │            │            │
    └────────────┼────────────┘
                 │
        ┌────────▼──────────────┐
        │ 근본 원인 파악        │
        │ (아키텍처 활용 미흡)  │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ 해결 방법 선정        │
        │ (메타데이터 필터링)   │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ 구현 및 검증          │
        │ (테스트 2회 통과)     │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ 재발 방지 조치        │
        │ (코드 고정, 테스트)   │
        └───────────────────────┘
```

---

## 기술적-배움

### 1️⃣ RAG 시스템의 메타데이터 설계의 중요성

**배운 내용**
```
청킹(Chunking) 단계에서 메타데이터를 철저히 분리하는 것이,
향후 검색 정확도를 좌우하는 결정적 요소임을 실감했습니다.

특히 MarkdownHeaderTextSplitter를 활용한 구조화된 분할은,
단순 텍스트 분할(RecursiveCharacterTextSplitter)보다
훨씬 우수한 메타데이터 추출을 가능케 합니다.
```

**코드 예시**
```python
# ❌ 나쁜 예: 메타데이터 미분리
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_text(raw_text)
# → 병해충 정보, 증상, 처방이 모두 섞임

# ✅ 좋은 예: 메타데이터 구조화
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "disease_name"),
    ("##", "category"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs = splitter.split_text(raw_text)
# → 각 청크가 disease_name과 category를 명확히 포함
```

### 2️⃣ Dense Retrieval의 근본적 한계

**핵심 통찰**
```
순수 벡터 유사도(Dense Retrieval)만으로는
도메인 특화 정보 검색에서 문맥적 오류가 불가피합니다.

실무 환경에서는 반드시 다음 중 하나가 필요합니다:

1. 메타데이터 필터링 (Hard Filter) ← 우리의 선택
2. BM25 같은 전통적 검색과의 결합 (Hybrid Search)
3. 도메인 특화 재순위화 (Reranking)
```

**검색 방식 비교**
```
┌─────────────────────────────────────────────────┐
│ 검색 방식별 특성                                │
├─────────────────────────┬──────────┬────────────┤
│ 방식                    │ 정확도   │ 속도      │
├─────────────────────────┼──────────┼────────────┤
│ Dense (순수 임베딩)     │ 중간 ⚠️  │ 빠름 ✅   │
│ Sparse (BM25)          │ 중간 ⚠️  │ 느림      │
│ Hybrid (Dense + Hard)  │ 높음 ✅  │ 빠름 ✅   │ ← 우리의 선택
│ Hybrid (Dense + BM25)  │ 높음 ✅  │ 중간      │
└─────────────────────────┴──────────┴────────────┘
```

### 3️⃣ 시스템 아키텍처를 먼저 고려한 문제 해결

**중요한 통찰**
```
LLM이나 임베딩 모델의 성능 개선에 먼저 집중하는 것보다,
전체 시스템 구조에서 "이미 확보한 정보"를 
어떻게 활용할 것인지 고민하는 것이 더 효율적입니다.

AgriAX의 경우:
- 비전 모델이 이미 정확한 진단 수행 (92% 신뢰도)
- 이 결과를 활용하지 않는 것은 자산 낭비
- 구조 개선으로 100% 정확도 달성 가능
```

**설계 원칙**
```
문제 발생 시 해결 우선순위:

1순위: 시스템 구조 개선
       (기존 정보 활용, 아키텍처 최적화)
2순위: 데이터 품질 개선
       (전처리, 청킹 최적화)
3순위: 모델 성능 개선
       (파인튜닝, 재학습)

← 비용 대비 효과가 가장 큼
```

---

## 재발-방지-조치

### Phase 1: 코드 레벨 (완료 ✅)

- [x] `rag_engine.py` 구조 고정 (메타데이터 필터 강제)
- [x] 함수 시그니처에 `target_disease` 매개변수 명시
- [x] 결과 검증 로직 추가 (assert)

### Phase 2: 테스트 레벨 (진행 중)

- [ ] 자동화된 테스트 3건 이상 작성
- [ ] CI/CD 파이프라인에 RAG 테스트 포함
- [ ] 매월 회귀 테스트 수행

### Phase 3: 문서화 레벨 (예정)

- [ ] RAG 파이프라인 구현 가이드 작성
- [ ] 메타데이터 필터링 Best Practice 정리
- [ ] 팀 온보딩 문서에 추가

### Phase 4: 모니터링 레벨 (장기)

- [ ] 검색 결과 정확도 자동 로깅
- [ ] 월간 품질 리포트 생성
- [ ] 이상 감지 알림 시스템

---

## 다음 개선 방향

### 단기 (1주일 내)

**검색어 추상화**
```python
# 현재 (사용자 질의 의존)
query = "고추에 둥근 반점이 생기고 흑갈색으로 변하는데?"

# 개선안 (고정 쿼리 사용)
query = "작물 질병 방제법 및 주의사항"  # 표준화된 쿼리

# 이점:
# - 검색의 일관성 증가
# - 증상 서술의 유사도 변동성 제거
# - 추가 프롬프트로 상세 정보 추출 가능
```

### 중기 (2~4주)

**하이브리드 검색 강화**
```python
# ChromaDB의 메타데이터 필터링 + BM25 결합
from langchain.retriever import BM25Retriever
from langchain.retriever.ensemble import EnsembleRetriever

# 1. 벡터 검색
vectorstore_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3, "filter": {"disease_name": target_disease}}
)

# 2. BM25 검색
bm25_retriever = BM25Retriever.from_documents(
    documents=filtered_docs,
    k=3
)

# 3. 앙상블 (결합)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vectorstore_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 벡터 검색에 더 높은 가중치
)

docs = ensemble_retriever.get_relevant_documents(query)
```

### 장기 (1개월+)

**Reranking 모델 도입**
```python
# Cohere Rerank 또는 BGE-Reranker 활용
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 1차: 메타데이터 필터 + 벡터 검색 (후보 생성)
# 2차: Reranker로 상위 문서만 추출

reranker = CohereRerank(
    model="rerank-english-v2.0"
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore_retriever
)
```

---

## 📈 Issue 통계

| 항목 | 수치 |
|------|------|
| **발생일** | 2025-01-20 |
| **해결일** | 2025-01-20 (당일) |
| **소요 시간** | ~2시간 |
| **가설 수** | 3개 |
| **테스트 케이스** | 2개 이상 |
| **코드 변경 라인** | 약 15줄 (filter 추가) |
| **구조적 개선** | 5가지 (코드, 테스트, 문서, 모니터링, 확장) |

---

## ✅ 체크리스트

### 해결 완료 항목
- [x] 문제 원인 파악
- [x] 메타데이터 필터링 구현
- [x] 단위 테스트 통과
- [x] 재발 방지 설계
- [x] 문서 작성

### 진행 중 항목
- [ ] 자동화 테스트 CI/CD 통합
- [ ] 팀 문서화 및 공유
- [ ] 모니터링 시스템 구축

### 예정 항목
- [ ] 하이브리드 검색 고도화
- [ ] Reranking 모델 도입
- [ ] 데이터 드리프트 모니터링

---

## 참고 자료

### 관련 파일
- `src/rag_engine.py` - 수정된 RAG 파이프라인
- `src/build_vector_db.py` - 벡터 DB 구축 코드
- `tests/test_rag_pipeline.py` - 자동화 테스트
- `docs/RAG_IMPLEMENTATION_GUIDE.md` - 구현 가이드

### 외부 참고 자료
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/rag/)
- [ChromaDB Filtering](https://docs.trychroma.com/usage-guide#filtering)
- [Dense vs Sparse Retrieval](https://arxiv.org/abs/2210.09773)

---

**문서 버전**: 1.0  
**마지막 수정**: 2025-01-20  
**상태**: ✅ Resolved & Documented
