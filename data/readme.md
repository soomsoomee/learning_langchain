# 데이터 가공 과정

1. `raw/tsla-20250930-gen.pdf` 에서 텍스트 추출해서 `data/processed/texts.json`에 저장
    - RecursiveCharacterTextSplitter 로 5000자씩 자름.
    - 동일한 페이지에서 나온 문서만 하나의 document가 될 수 있음. 
2. `raw/tsla-20250930.html` 에서 테이블 추출해서 `data/processed/tables.json`에 저장
    - table html을 마크다운으로 바꿔서 LLM을 통해 description 생성
3. texts.json과 table.json의 description을 multi-lingual 모델인 BAAI/bge-m3로 임베딩 (8000토큰까지 가능)
4. chroma db에 저장
