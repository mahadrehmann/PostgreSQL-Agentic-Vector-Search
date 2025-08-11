# # ingest_embeddings.py

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import PGVector
# from langchain.schema import Document
# from dotenv import load_dotenv
# import os

# # Load env variables
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# connection_string = os.getenv("DATABASE_URL")  # e.g. 'postgresql://postgres:pass@localhost:5432/demo'

# # 1. Load and split the document
# loader = TextLoader(file_path="sherlock.txt")
# data = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", " ", "."],
#     chunk_size=200,
#     chunk_overlap=50,
# )
# chunks = splitter.split_text(data[0].page_content)
# docs = [Document(page_content=chunk) for chunk in chunks]

# # 2. Set up embeddings
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_key=api_key
# )

# # 3. Store embeddings in PostgreSQL via PGVector
# vectorstore = PGVector.from_documents(
#     documents=docs,
#     embedding=embedding_model,
#     connection_string=connection_string,
#     collection_name="sherlock_collection",
#     use_jsonb=True,
# )

# print("Ingestion completed: document chunks embedded and stored.")

# ingest_embeddings.py

from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.schema import Document

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
connection_string = os.getenv("DATABASE_URL")
collection_name = "sherlock_collection"

# 1. Load and split the document
loader = TextLoader(file_path="data/sherlock.txt")
docs_raw = loader.load()
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", "."],
    chunk_size=200,
    chunk_overlap=50,
)
chunks = splitter.split_text(docs_raw[0].page_content)
docs = [Document(page_content=chunk) for chunk in chunks]

# 2. Set up embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# 3. Store embeddings in PostgreSQL with correct signature
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    connection=connection_string,
    collection_name=collection_name,
    pre_delete_collection=True,
    use_jsonb=True
)

print("Ingestion completed: document chunks embedded and stored correctly.")
