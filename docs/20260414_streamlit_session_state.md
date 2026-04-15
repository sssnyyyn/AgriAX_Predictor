# 🛠 AgriAX Predictor v2.0 - 트러블슈팅 기록

**문서 작성일**: 2026년 4월 14일
**프로젝트**: AgriAX Predictor v2.0 LLM 고도화
**담당자**: [개인 프로젝트]

---

## 📑 목차

1. [Issue #1: Streamlit 탭 이동 시 분석 데이터 초기화(State Reset) 현상](#issue-1)
2. [해결 방법론 정리](#해결-방법론)
3. [기술적 배움](#기술적-배움)
4. [재발 방지 조치](#재발-방지-조치)

---

<a id="issue-1"></a>
## Issue #1: Streamlit 탭 이동 시 분석 데이터 초기화(State Reset) 현상

### 📌 Issue 개요

| 항목 | 내용 |
|------|------|
| **Issue Title** | 탭 간 이동 및 버튼 상호작용 시 이전 분석 데이터 증발 현상 |
| **Severity** | 🔴 **Critical** - 파이프라인 단절로 인한 서비스 기능 마비 |
| **Status** | ✅ **RESOLVED** |
| **Affected Component** | 대시보드 UI (Tab 1 진단 ↔ Tab 4 재무 분석 연동) |
| **File Location** | `AgriAX.py` |

---

### 🔍 상황 분석

#### 발생 환경
- **환경**: 로컬 개발 환경 (Python 3.10, Streamlit 1.28+)
- **시점**: 재무적 타격 예측 모듈 연동 및 UI 테스트 단계 (v2.0 개발 3일차)
- **데이터**: Vision AI 진단 결과 및 RAG 처방 데이터

#### 문제 증상

**사용자 플로우 (문제 상황)**
```text
1. [Tab 1] 이미지 업로드 후 '멀티모달 통합 분석 실행' 클릭
2. 진단 결과 및 처방전 정상 출력됨 (변수: diagnosis 생성)
3. [Tab 4] 비즈니스 ROI 탭으로 이동
4. 면적 입력 후 '재무 시나리오 분석 실행' 클릭
5. ❌ 로딩이 완료되기 전 화면이 새로고침 됨
6. ❌ Tab 4에 "분석을 먼저 실행해 주십시오" 경고 발생
7. ❌ Tab 1로 돌아가면 이전 분석 결과가 모두 날아가고 초기 상태로 돌아감
```

#### 영향 범위

```
Vision 모델(진단)
    ↓
RAG 검색(처방)
    ↓  (데이터 단절) 💥
LLM 재무 분석(Tab 4 연동 불가)
```

**결과**:
- ❌ 진단-처방-분석-리포트로 이어지는 End-to-End 파이프라인 단절
- ❌ 사용자가 모든 탭에서 분석을 처음부터 다시 해야 하는 치명적 UX 결함
- ❌ 프로덕션 배포 불가능 상태

---

### 🧠 원인 분석 프로세스

#### 가설 설정

#,가설,검증 방법,결과
1,Tab UI Unmount 시 변수 메모리 해제,탭 이동 후 st.write() 로깅,❌ 기각
2,LLM 로딩 지연에 따른 비동기 충돌,타임아웃/슬립 지연 테스트,❌ 기각
3,Streamlit 고유의 전체 스크립트 재실행,버튼 클릭 전후 실행 로그 추적,✅ 채택

#### 상세 검증 과정
**[검증 1] Tab 이동 시 메모리 해제 여부**
```python
# 탭 이동만 했을 때는 데이터가 남아있는지 확인
결과: Tab 1 분석 후 단순히 Tab 4를 누른 직후에는 화면에 조건문이 통과되어 버튼 UI가 보임.

결론: 탭 UI 자체의 Unmount가 변수를 날리는 것은 아님 (가설 1 기각)
```

**[검증 2] Streamlit 실행 메커니즘 분석** ⭐
```python
버튼 클릭 이벤트 발생 시 프레임워크 동작 방식 추적:
1. 사용자가 Tab 4의 '재무 분석' 버튼 클릭
2. Streamlit이 AgriAX.py 파일의 첫 줄부터 다시 코드 실행(Rerun)
3. 코드 진행 중 Tab 1의 '통합 분석' 버튼 블록 도달
   -> 현재 클릭된 상태가 아니므로 내부 코드(diagnosis = ...) 스킵
4. Tab 4 코드 블록 도달
   -> diagnosis 변수가 메모리에 할당된 적이 없으므로 NameError 또는 로직 우회

결론: 선언형 UI 프레임워크의 Top-to-bottom Execution 특성 (가설 3 채택)
```

#### 원인의 근본

**기술적 근본 원인**
```
일반 웹 어플리케이션(React 등):
상태(State)가 컴포넌트 생명주기와 함께 유지됨.

Streamlit 어플리케이션:
사용자 상호작용 발생 시 스크립트 전체 재실행.
로컬 변수(Local Variables)는 매 실행마다 초기화됨.

┌─────────────────────────────────────────────┐
│  실행 사이클 1 (Tab 1 버튼 클릭)            │
│  → diagnosis 변수 생성 (성공)              │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  실행 사이클 2 (Tab 4 버튼 클릭)            │
│  → 위에서부터 다시 실행                    │
│  → Tab 1 버튼은 False 상태                 │
│  → diagnosis 변수 미생성 (증발)            │
└─────────────────────────────────────────────┘
```

---

### 💡 해결 방법

#### 수정 전 코드

```python
# AgriAX.py (문제 코드)

# [Tab 1] 지역 변수로 할당
if st.button("멀티모달 통합 분석 실행"):
    # ❌ 실행 사이클이 끝나면 사라지는 휘발성 변수
    diagnosis = get_disease_info(pred_idx)

# [Tab 4] 휘발성 변수에 의존
if 'diagnosis' in locals() and diagnosis["name"] != "정상":
    if st.button("재무 시나리오 분석 실행"):
        analyzer.generate_roi_scenario(diagnosis["name"], area_input)
```

**문제점**
- locals()에 의존하여 런타임 주기 변경 시 데이터 유실
- 프레임워크의 상태 비저장(Stateless) 특성을 역행하는 구조

---

#### 수정 후 코드

```python
# AgriAX.py (해결된 코드)

# [Tab 1] 전역 세션 저장소 활용
if st.button("멀티모달 통합 분석 실행"):
    diagnosis = get_disease_info(pred_idx)

    # ✅ Step 1: session_state에 안전하게 보관 (메모리 고정)
    st.session_state['diagnosis'] = diagnosis

# [Tab 4] 세션 저장소에서 호출
# ✅ Step 2: session_state 존재 여부 및 무결성 확인
if 'diagnosis' in st.session_state and st.session_state['diagnosis']["name"] != "정상":
    current_diagnosis = st.session_state['diagnosis']

    if st.button("재무 시나리오 분석 실행"):
        analyzer.generate_roi_scenario(current_diagnosis["name"], area_input)
```

**개선사항**
- Streamlit의 네이티브 상태 관리 객체(st.session_state) 도입
- 탭 이동 및 버튼 클릭 시 Rerun이 발생해도 데이터 영속성 유지 보장
- locals() 의존성 완벽 제거

---

#### 선택 이유

방법,장점,단점,적용 여부
st.session_state 사용,"내장 기능으로 구현 빠름, 메모리 관리 용이",세션 종료 시 휘발,✅ 선택
파일(JSON/DB) 입출력,영구 보존 가능,"I/O 오버헤드, 5일 프로젝트에 오버엔지니어링",❌
Redis 외부 캐시,확장성 우수,1인 개발 인프라 구축 시간 과다,❌
URL 파라미터 활용,공유 용이,데이터 크기 제한 (JSON 객체 담기 불가),❌

**최종 선택 근거**
```
1인 개발 체제의 5일(Day 3)이라는 빡빡한 타임라인 속에서:
1️⃣ 프레임워크의 기본 철학(Native Way)을 따르는 것이 가장 안정적임
2️⃣ 외부 DB나 캐시 의존성을 추가하는 것은 불필요한 공수 낭비
3️⃣ 시스템 재시작 전까지 유지되는 Session 상태로 파이프라인 연결 요건 충족
```

---

### 📊 해결 결과

#### 검증 방법

**[테스트 1] 크로스 탭 데이터 유지 테스트**
1. Tab 1에서 고추 탄저병 이미지 분석 실행 (성공)
2. Tab 2, Tab 3 클릭하여 이동 후 Tab 4 진입
3. '재무 시나리오 분석 실행' 버튼 클릭
4. 결과: 화면 새로고침 없이 Ollama 추론 성공 및 UI 렌더링 유지 ✅
5. 다시 Tab 1로 이동 시 진단 결과 원형 보존 확인 ✅

#### 성과

항목,Before,After
데이터 유지율,0% (상호작용 시 증발),100%
사용자 이탈 리스크,매우 높음,0%
코드 구조,휘발성,영속성 (Session Base)
파이프라인 연속성,⚠️ 단절,✅ 완전 연결

---

### 🔄 재발 방지 조치

#### 1단계: 코드 레벨 설계 고정

**파일**: `AgriAX.py` (전역 컨벤션 확립)

```python
# 향후 개발될 모든 탭간 데이터 교환은 아래 규칙을 준수

# [저장 규칙] Rerun에 영향받지 않아야 하는 주요 데이터는 무조건 세션 등록
st.session_state['report_data'] = final_report

# [호출 규칙] 일반 변수가 아닌 세션 딕셔너리에서 안전하게 get
saved_report = st.session_state.get('report_data', None)
if saved_report:
    # 렌더링 로직
```

**핵심 규칙**
```
📋 Streamlit 상태 관리 원칙

[필수 규칙]
1. 사용자 입력값 외의 연산 결과는 `st.session_state`에 적재
2. 타 모듈/탭에서 데이터 참조 시 `if key in st.session_state` 방어 로직 필수

[금지사항]
- ❌ `locals()`나 `globals()`를 활용한 편법 데이터 전달
- ❌ 함수 외부의 글로벌 변수 재할당
```

#### 2단계: Streamlit 통합 시 강제

**파일**: `src/insight_report_generator.py`

```python
# 4일차 리포트 모듈 연동 시, 개별 데이터 쿼리 대신
# 세션에 누적된 전체 파이프라인 컨텍스트를 한 번에 주입하는 구조로 고정
def generate_final_report():
    if 'diagnosis' not in st.session_state:
        st.warning("초기 진단 데이터가 없습니다.")
        return
    # 세션 데이터 통과 시에만 LLM 체이닝 실행
```

---

## 해결-방법론

### 문제 해결 프로세스 (What I Learned)

```
┌────────────────────────────────────────────────┐
│ 문제 발생                                      │
│ (탭 간 이동 시 데이터 증발 현상)              │
└────────────────┬─────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │ 가설 수립 (3개)   │ ← 선언형 프레임워크 한계 의심
        └────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
  가설 1       가설 2       가설 3
UI Unmount  비동기 충돌   Top-to-Bottom
  메모리해제                  재실행 (Rerun)
    ❌           ❌           ✅
   기각         기각         채택
    │            │            │
    └────────────┼────────────┘
                 │
        ┌────────▼──────────────┐
        │ 근본 원인 파악        │
        │ (Stateless 설계 특성) │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ 해결 방법 선정        │
        │ (st.session_state)    │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ 구현 및 검증          │
        │ (탭 간 데이터 유지)   │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ 재발 방지 조치        │
        │ (상태 관리 컨벤션)    │
        └───────────────────────┘
```

---

## 기술적-배움

### 1️⃣ RAG 선언형 UI 프레임워크의 생명주기 이해

**배운 내용**
```
Streamlit이나 Gradio 같은 데이터 앱 프레임워크는
전통적인 이벤트 루프(Event Loop) 기반의 GUI 프로그래밍과 완전히 다릅니다.

이벤트(버튼 클릭)가 발생할 때마다 해당 콜백만 실행되는 것이 아니라,
스크립트의 처음부터 끝까지 전체가 재실행(Rerun)된다는 패러다임 전환을
명확히 이해하는 계기가 되었습니다.
```

### 2️⃣ 로컬 변수 vs 세션 변수의 차이

**핵심 통찰**
```
스크립트 기반 환경에서 변수의 생존 범위(Scope) 설계:

- 로컬 변수(Local Variable): 단일 렌더링 사이클 내에서만 유효. 화면에 한 번 그리고 버릴 임시 계산값에 적합.
- 세션 변수(Session State): 사용자 세션이 닫히기 전까지 지속. 탭 간 통신이나 다단계 마법사(Wizard) 형태의 폼 구축에 필수적.
```

### 3️⃣ 1인 개발 환경에서의 빠른 디버깅 원칙

**설계 원칙**
```
명시적인 에러 메시지(Traceback)가 없는 논리적 오류를 디버깅할 때,
코드 라인을 고치기 전 프레임워크의 공식 문서(Architecture Document)를
먼저 확인하는 것이 1인 개발의 시간을 극적으로 단축시킴을 깨달았습니다.
```

---

## 재발-방지-조치

### Phase 1: 코드 레벨 (완료 ✅)

- [x] 전역 상태 공유를 위한 session_state 컨벤션 확립
- [x] 탭 4 데이터 호출 로직 리팩토링
- [x] 방어 코드(if key in st.session_state) 의무화

### Phase 2: UX 레벨 (진행 중)

- [ ] 데이터가 없을 때 타 탭의 버튼 비활성화(disabled=True) 적용

- [ ] Rerun 시 시각적 끊김을 방지하기 위한 캐싱(@st.cache_data) 추가

### Phase 3: 아키텍처 레벨 (예정)

- [ ] Streamlit on_click 콜백 함수 기반의 이벤트 지향 아키텍처로 점진적 마이그레이션

---

## 다음 개선 방향

### 단기 (1주일 내)

**사용자 동선 제어 (UX 개선)**
```python
# 무의미한 에러 경고를 띄우기 전, 버튼 자체를 잠그는 방식

has_data = 'diagnosis' in st.session_state

if st.button("재무 분석 실행", disabled=not has_data):
    # 로직 실행
```

### 중기 (2~4주)

**콜백(Callback) 함수 활용**
```python
# Top-to-Bottom 재실행의 오버헤드를 줄이기 위한 콜백 아키텍처

def process_analysis():
    st.session_state.diagnosis = get_disease_info(pred_idx)

# 버튼 클릭 시 전체 코드를 돌기 전, 콜백을 우선 실행
st.button("통합 분석", on_click=process_analysis)
```

---

## 📈 Issue 통계

항목,수치
발생일,2026-04-14
해결일,2026-04-14 (당일)
소요 시간,~1.5시간
가설 수,3개
테스트 케이스,1개 (End-to-End 플로우)
코드 변경 라인,약 10줄 (session_state 교체)
구조적 개선,"3가지 (상태관리 컨벤션, UX제어, 콜백 준비)"

---

## ✅ 체크리스트

### 해결 완료 항목
- [x] 문제 원인 파악 (Rerun Lifecycle)
- [x] session_state 기반 상태 유지 구현
- [x] 탭 간 플로우 연속성 검증
- [x] 문서 작성

### 진행 중 항목
- [ ] 버튼 비활성화 방어 로직 추가
- [ ] 파이프라인 3단계(리포트) 연동 준비

---

## 참고 자료

### 관련 파일
- `AgriAX.py` - 상태 관리 로직이 적용된 메인 앱
- `src/financial_analyzer.py` - 연동 대상 모듈

### 외부 참고 자료
- Streamlit Session State Documentation
- Streamlit App Architecture & Execution Flow

---

**문서 버전**: 1.0
**마지막 수정**: 2026-04-15
**상태**: ✅ Resolved & Documented
