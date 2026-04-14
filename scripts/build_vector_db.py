"""
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 텍스트 청킹 (이전 단계와 동일)
file_path = "data/md/agri_disease_batch_1.md"
with open(file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [
    ("#", "disease_name"),
    ("##", "category")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = markdown_splitter.split_text(markdown_document)

# 2. 임베딩 모델 로드 (한국어 특화 모델)
model_name = "jhgan/ko-sroberta-multitask"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 3. ChromaDB 벡터 저장소 생성 및 데이터 저장
persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"데이터베이스 구축 완료: {len(chunks)}개의 문서가 {persist_directory}에 저장되었습니다.\n")

# 4. 검색 테스트 (Retriever)
query = "고추에 둥근 반점이 생기고 흑갈색으로 변하는데 어떻게 해야 해?"

print(f"질의: {query}")
print("--- 검색 결과 ---")

# k=2: 가장 유사도가 높은 상위 2개의 청크를 가져옴
docs = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(docs):
    print(f"\n[순위 {i+1}]")
    print(f"메타데이터: {doc.metadata}")
    print(f"내용: {doc.page_content}")
"""


import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 문서 로드 및 청킹
file_path = "data/md/agri_disease_batch_1.md"
with open(file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [
    ("#", "disease_name"),
    ("##", "category")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = markdown_splitter.split_text(markdown_document)

# 2. 임베딩 모델 로드
model_name = "jhgan/ko-sroberta-multitask"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 3. ChromaDB 저장소 연결 및 데이터 저장
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)
print(f"데이터베이스 로드 완료: {persist_directory}")

# 4. 메타데이터 필터링 검색 테스트
# 실제 서비스에서는 비전 모델(agriax.py)이 예측한 질병명이 target_disease 변수에 할당됩니다.
target_disease = "고추 탄저병"
query = "둥근 반점이 생기고 흑갈색으로 변하는데 어떻게 해야 해?"

print(f"\n[검색 조건] 대상 질병: {target_disease} / 질의: {query}")
print("-" * 50)

# filter 인자를 통해 disease_name이 일치하는 청크만 검색 대상으로 한정합니다.
docs = vectorstore.similarity_search(
    query=query,
    k=2,
    filter={"disease_name": target_disease}
)

for i, doc in enumerate(docs):
    print(f"[순위 {i+1}]")
    print(f"메타데이터: {doc.metadata}")
    print(f"내용: {doc.page_content}\n")
