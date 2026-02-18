# rag-chatbot

로컬/오프라인 환경에서 기준문서를 기반으로 답변하는 RAG 챗봇 프로젝트입니다.  
(문서 전처리 → 청크/메타데이터 생성 → 임베딩/인덱싱 → 검색 기반 답변)

## 구조
- `data/raw/` : 원본 기준문서(PDF/MD 등)
- `data/processed/` : 최종 청크 모음(`corpus.jsonl`)
- `models/` : 로컬 모델 파일(임베더/리랭커/LLM gguf)
- `indexes/` : 검색 인덱스(FAISS/BM25 등)
- `configs/` : 팀 공용 설정(app/models/rag/prompts/logging)
- `src/` : ingest, embed, rag, app(cli/server), eval 코드

## 빠른 흐름
1) 문서 넣기  
- `data/raw/`에 기준문서 추가

2) 전처리(ingest)  
- 텍스트 추출/헤딩 파싱/청킹 후 `data/processed/corpus.jsonl` 생성 확인

3) 인덱스 생성  
- `corpus.jsonl` → 임베딩 → `indexes/vector/` 생성

4) 실행  
- CLI로 질문 → 검색 → 답변(+인용)

## 설정
- 공용 설정: `configs/*.yaml`
- 개인 경로/환경: `.env.example`을 복사해 `.env`로 사용(`.env`는 커밋하지 않음)

## 모델/데이터 관리
- `models/`, `data/`, `indexes/`는 용량이 크므로 기본적으로 git에 올리지 않습니다.
- 다운로드/전처리/인덱스 생성은 `scripts/`의 스크립트로 재현 가능하게 유지합니다.


### ingest 실행 예시
- `python -m src.ingest.build_corpus --data-raw-dir data/raw --output data/processed/corpus.jsonl`
- 산출물 `corpus.jsonl`은 각 줄이 JSON이며, 최소 메타데이터(`doc_id`, `section_path`, `heading`, `chunk_id`, `chunk_index`, `page_start`, `page_end`, `chunk_tokens`)를 포함합니다.
