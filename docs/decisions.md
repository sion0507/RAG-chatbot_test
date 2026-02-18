# docs/decisions.md

## Confirmed(확정)
### 목표/제약
- 인터넷 없이 로컬 PC에서 운영 가능한 RAG 챗봇 구현을 목표로 합니다.
- GPU 자원이 열악하므로, GPU는 가능하면 LLM 추론에 우선 배정하고 임베딩/리랭킹은 CPU 우선으로 설계합니다.
- 답변은 기준 문서 근거에 기반해야 하며, 근거 없는 단정(환각)을 최소화합니다.

### 추론 엔진
- LLM 추론 엔진은 llama.cpp를 사용합니다.
- 오프라인 운영을 위해 GGUF 등 로컬 로딩 가능한 형식을 우선 고려합니다.

### 임베딩 모델
- 임베딩 모델은 `dragonkue/multilingual-e5-small-ko-v2`를 사용합니다.

### 핵심 라이브러리(현재 기준)
- API/서버: `fastapi`, `uvicorn`
- 유틸: `requests`, `numpy`, `tqdm`
- 문서 파싱: `pymupdf`
- 임베딩: `sentence-transformers`, `torch`
- 검색: `faiss-cpu`, `rank-bm25`
- (선택) `python-dotenv`, `python-docx`
- (대체) Windows 등 환경 이슈 시 `faiss-cpu`대신  `hnswlib`를 고려

### 청킹/메타데이터(최소 필드)
- 확정 메타데이터 필드:
  - `doc_id`
  - `section_path`(헤딩 기반 섹션 경로)
  - `heading`
  - `chunk_id`, `chunk_index`
  - `page_start`, `page_end`
  - `chunk_tokens`
- `section_path`는 “문서 내 위치를 안정적으로 지칭”하기 위한 필드로 유지합니다.

## Pending(미확정/추후 결정)
### LLM 모델 구체 선택
- 어떤 LLM(예: Qwen 계열/다른 계열), 파라미터 크기(7B/14B 등), 양자화 레벨은 아직 확정되지 않았습니다.
- 확정 시: 모델명/버전/포맷(GGUF)/라이선스/권장 컨텍스트 길이를 함께 기록합니다.

### 리랭커 모델
- 후보로 `dragonkue/bge-reranker-v2-m3-ko`를 고려하고 있으나 “최종 확정”은 아닙니다.
- CPU 성능/지연/품질 트레이드오프를 보고 확정합니다.

### 인덱스 저장/캐시 고정 방식
- HF 캐시 경로 고정(HF_HOME 등), 인덱스 파일 저장 위치/백업 정책은 프로젝트 운영 환경에 맞춰 확정합니다.

### 인용(citation) 포맷
- 인용은 “doc_id + section_path + page_start/end + chunk_id(or index)” 조합으로 생성할 수 있으며,
  - 문자열 포맷은 답변 UI/출력 요구에 맞춰 추후 확정합니다.

## 규칙(변경 관리)
- 위 Confirmed 항목을 변경하는 PR은 `docs/architecture.md` 및 본 파일(`docs/decisions.md`)을 같이 업데이트합니다.
- Pending 항목이 확정되면 Confirmed로 승격하고, 근거(실험 결과/운영 요구)를 간단히 남깁니다.
