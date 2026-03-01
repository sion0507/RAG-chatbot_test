# Phase B 구현 계획: 리랭커 통합

## 0) 전제
- Phase A(하이브리드 검색 후보 생성)는 완료된 상태를 전제로 합니다.
- BM25 토크나이저 교체는 별도 작업으로 분리하고, 본 문서는 **리랭킹 파이프라인 통합**에 집중합니다.

## 1) 목표
- BM25 + FAISS 하이브리드 검색으로 1차 후보 **10~20개**를 구성합니다.
- 1차 후보(`rerank_candidates`)를 크로스 인코더 리랭커로 재정렬합니다.
- 최종 Top-N은 **리랭커 점수 기준 상위 3~5개**를 사용합니다.
- 근거 메타데이터(`doc_id`, `section_path`, `page_start/end`, `chunk_id`)는 손실 없이 전달합니다.

## 2) 구현 범위(코드 단위)

### 2.1 `src/rag/reranker.py` 신설
- 역할: 질의 + 후보 청크를 입력받아 리랭킹 점수를 계산하고 정렬 결과를 반환.
- 제안 인터페이스:
  - `RerankCandidate` (dataclass)
    - `chunk: ChunkRecord`
    - `retrieval_score: float`
    - `rerank_score: float`
  - `CrossEncoderReranker` (class)
    - `__init__(model_name_or_path: str, device: str = "cpu", batch_size: int = 8)`
    - `rerank(query: str, candidates: list[RetrievalCandidate], top_n: int) -> list[RerankCandidate]`
- 동작 원칙:
  - CPU 기본 장치 사용(설정으로만 변경 가능)
  - 입력 길이 제한/트렁케이션은 모델 토크나이저 기본 정책을 사용
  - 빈 후보 입력 시 빈 리스트 반환

### 2.2 `src/rag/retriever.py` 확장
- 현재 `retrieve()` 결과(`RetrievalResult`)를 리랭커 입력으로 쓰고,
  선택적으로 `rerank()` 호출을 수행하는 오케스트레이션 메서드 추가.
- 예시:
  - `retrieve_for_rerank(query, rerank_top_k=15)` → 하이브리드 1차 후보(권장 10~20개)
  - `retrieve(query, reranker, final_top_n=5)`
    - `reranker`는 필수 입력으로 강제
    - `rerank_candidates`를 리랭킹해 최종 Top-N(권장 3~5) 반환

### 2.3 설정 반영
- `configs/models.yaml`
  - `reranker.model_name_or_path`
  - `reranker.device` (기본 `cpu`)
  - `reranker.batch_size`
- `configs/rag.yaml`
  - `rerank.rerank_top_k` (권장 10~20 범위, 기본 15)
  - `rerank.final_top_n` (권장 3~5 범위, 기본 5)
  - `context.max_chunks_in_context` (권장 3~5, 기본 5)
- 설정값은 코드 하드코딩 없이 주입되도록 유지

### 2.4 앱 레이어 연결(다음 단계)
- 채팅 엔드포인트/CLI가 `HybridRetriever + CrossEncoderReranker`를 초기화해 사용하도록 연결
- 최종 응답 페이로드에 리랭크 점수와 인용 메타데이터를 포함할 수 있게 스키마 확장

## 3) 오류/운영 처리 기준
- 리랭커 모델 로딩/실행 실패 시:
  - 예외를 상위로 전달해 요청 실패로 처리(필수 단계 비활성화 은닉 금지)
- 후보 수가 `rerank_top_k`보다 적은 경우:
  - 가능한 후보만 리랭킹하고 정상 반환
- 오프라인 운영:
  - 로컬 경로 모델 우선, 네트워크 다운로드 전제 금지

## 4) 테스트 계획

### 4.1 단위 테스트(`tests/test_reranker.py`)
- 더미 리랭커/모킹으로 점수 내림차순 정렬 검증
- `top_n` 컷오프 검증
- 빈 후보 입력 검증

### 4.2 통합 테스트(`tests/test_retrieval_pipeline.py` 확장)
- 기존 phase A 테스트 유지
- 리랭커 주입 시 최종 순서가 retrieval 점수와 달라지는 케이스 검증
- 리랭커 예외 발생 시 예외 전파(실패 처리) 검증

## 5) 작업 순서(권장)
1. `reranker.py` + 단위 테스트 추가
2. `retriever.py` 오케스트레이션 확장 + 통합 테스트 보강
3. 설정 파일(`models.yaml`, `rag.yaml`) 반영
4. 앱 레이어 연결(별도 커밋)

## 6) BM25 토크나이저 교체와의 경계
- BM25 토크나이저 변경은 **1차 후보 품질 개선** 작업으로 분리 유지
- Phase B 완료 기준은 “후보 생성 이후 리랭킹 로직이 안정적으로 동작”하는지에 둡니다.
- 즉, 토크나이저 변경 전/후 모두 리랭킹 모듈은 동일 인터페이스로 재사용 가능해야 합니다.

## 7) 이번 가이드의 설정 우선순위
- 문서/코드/기존 config와 수치가 다를 경우, 현재 운영 가이드 우선순위는 아래와 같이 둡니다.
  - 1차 후보 수: 10~20
  - 리랭킹 후 LLM 컨텍스트 전달 청크: 3~5
