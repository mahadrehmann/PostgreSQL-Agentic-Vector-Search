from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
connection_string = os.getenv("DATABASE_URL")  # e.g. 'postgresql+psycopg2://postgres:pass@localhost:5432/demo'

# 1. Load and split the document
loader = TextLoader(file_path="sherlock.txt")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", "."],
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_text(data[0].page_content)

docs = [Document(page_content=chunk) for chunk in chunks]

# 2. Set up embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# 3. Initialize PGVector store and add the document chunks
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    connection_string=connection_string,
    collection_name="sherlock_collection",
    use_jsonb=True,
)

# 4. Query the vector store
def query_store(query_text):
    results = vectorstore.similarity_search(query_text, k=3)
    return [doc.page_content for doc in results]

query = "the woman"
results = query_store(query)

print("Top 3 Matching Chunks:\n")
for i, res in enumerate(results, start=1):
    print(i, res)
