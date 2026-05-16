from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. 저장한 마크다운 파일 읽기
file_path = "data/md/agri_disease_batch_1.md"
with open(file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()

# 2. 분할 기준 설정
# '#'은 질병명(disease_name)으로, '##'은 카테고리(category) 메타데이터로 매핑
headers_to_split_on = [
    ("#", "disease_name"),
    ("##", "category")
]

# 3. 분할기 초기화 및 실행
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

# 4. 결과 출력 및 확인
print(f"총 {len(md_header_splits)}개의 청크(Chunk)로 분할되었습니다.\n")

for i, chunk in enumerate(md_header_splits):
    print(f"--- Chunk {i+1} ---")
    print(f"메타데이터: {chunk.metadata}")
    print(f"내용: {chunk.page_content}")
    print("-" * 50)
