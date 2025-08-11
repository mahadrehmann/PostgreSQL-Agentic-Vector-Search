# # agent_query_flow.py (fixed)
# from dotenv import load_dotenv
# import os
# import json

# # phi / phidata agent imports
# from phi.agent import Agent
# from phi.model.openai import OpenAIChat

# # LangChain OpenAI embeddings and PGVector wrapper (adjust import if you use a different package)
# from langchain_openai import OpenAIEmbeddings
# from langchain_postgres.vectorstores import PGVector

# load_dotenv()
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# DB_URL = os.getenv("DATABASE_URL")
# COLLECTION = "sherlock_collection"   # must match your ingest

# # Embedding function
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_KEY)

# # Connect to an existing PGVector index (no inserts) -- adjust kwarg name if your PGVector expects different arg
# try:
#     vectorstore = PGVector.from_existing_index(
#         embedding=embedding_model,
#         collection_name=COLLECTION,
#         connection=DB_URL,            # if this fails try connection_string=DB_URL
#     )
# except TypeError:
#     # fallback alternative kwarg name
#     vectorstore = PGVector.from_existing_index(
#         embedding=embedding_model,
#         collection_name=COLLECTION,
#         connection_string=DB_URL,
#     )

# # --- Phidata (phi) agents ---
# metadata_agent = Agent(
#     name="MetadataCheckerAgent",
#     model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
#     instructions=[
#         "Given the short user query below, answer ONLY 'YES' if the DB is likely to contain relevant documents, otherwise answer 'NO'. Be concise."
#     ],
#     markdown=False,
# )

# db_agent = Agent(
#     name="DBAccessAgent",
#     model=OpenAIChat(id="gpt-4o-mini-2024-07-18"),
#     instructions=["You are given some document chunks below. Summarize or return them to the user as-is depending on the user prompt."],
#     markdown=True,
# )


# def runresponse_to_text(run_resp) -> str:
#     """
#     Safely extract a textual response from a Phidata RunResponse object.
#     Returns a string (possibly empty) for downstream use.
#     """
#     if run_resp is None:
#         return ""

#     # 1) content is the primary field
#     content = getattr(run_resp, "content", None)
#     if content:
#         # if content is already a string, return it
#         if isinstance(content, str):
#             return content.strip()
#         # if content is JSON-like (dict/list), dump it to readable string
#         try:
#             return json.dumps(content, indent=2, ensure_ascii=False)
#         except Exception:
#             return str(content)

#     # 2) sometimes responses come in 'messages' list
#     messages = getattr(run_resp, "messages", None)
#     if messages:
#         try:
#             parts = []
#             for m in messages:
#                 # message may be pydantic Message with .content or a dict
#                 c = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
#                 if c:
#                     parts.append(c)
#             if parts:
#                 return "\n".join(parts).strip()
#         except Exception:
#             # last resort
#             return str(messages)

#     # 3) fallback to whole object string
#     return str(run_resp)


# def check_metadata_with_vector(query: str, threshold_hits: int = 1) -> bool:
#     # quick deterministic metadata check = do we get any near neighbors for the query?
#     try:
#         hits = vectorstore.similarity_search(query, k=3)  # returns list[Document]
#         return len(hits) >= threshold_hits
#     except Exception as e:
#         print("Vectorstore check failed:", e)
#         return False


# def get_top_chunks(query: str, k: int = 3):
#     return vectorstore.similarity_search(query, k=k)


# def main():
#     user_query = input("Enter your query: ").strip()
#     if not user_query:
#         print("Empty query. Exiting.")
#         return

#     # 1) Ask the metadata agent (LLM) for a quick YES/NO
#     try:
#         meta_run = metadata_agent.run(user_query)
#         meta_text = runresponse_to_text(meta_run)
#     except Exception as e:
#         print("Metadata agent failed:", e)
#         meta_text = ""

#     decision_text = (meta_text or "").strip().upper()
#     # 2) Deterministic fallback (vector-space)
#     found = check_metadata_with_vector(user_query)

#     if decision_text == "YES" or found:
#         try:
#             chunks = get_top_chunks(user_query, k=3)
#             content = "\n\n---\n\n".join([c.page_content for c in chunks])
#             # 3) Hand retrieved content to DB agent for formatting/summarizing
#             db_input = f"User query: {user_query}\n\nRetrieved chunks:\n{content}\n\nReturn the most relevant chunks."
#             db_run = db_agent.run(db_input)
#             db_text = runresponse_to_text(db_run)
#             print("\n===== ANSWER =====\n")
#             print(db_text)
#         except Exception as e:
#             print("Error retrieving or processing chunks:", e)
#     else:
#         print("No relevant documents found in the DB.")


# if __name__ == "__main__":
#     main()


# agent_query_flow_debug.py
from dotenv import load_dotenv
import os
import json
import textwrap
import traceback

from phi.agent import Agent
from phi.model.openai import OpenAIChat

from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.schema import Document

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
COLLECTION = "sherlock_collection"

# Embedding function
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_KEY)

# Connect to existing vector index (try both kwarg names)
print("Connecting to PGVector index...")
try:
    vectorstore = PGVector.from_existing_index(
        embedding=embedding_model,
        collection_name=COLLECTION,
        connection=DB_URL,
    )
    print("Connected using kwarg 'connection'")
except Exception as e:
    print("Failed 'connection' attempt:", type(e).__name__, e)
    try:
        vectorstore = PGVector.from_existing_index(
            embedding=embedding_model,
            collection_name=COLLECTION,
            connection_string=DB_URL,
        )
        print("Connected using kwarg 'connection_string'")
    except Exception as e2:
        print("Failed to connect to PGVector. Aborting. Error:")
        traceback.print_exc()
        raise

# --- Agents with timeouts (set short timeout to fail fast) ---
MODEL_ID = "gpt-4o-mini-2024-07-18"
agent_model_kwargs = {"timeout": 30, "max_retries": 1}  # 30s timeout

metadata_agent = Agent(
    name="MetadataCheckerAgent",
    model=OpenAIChat(id=MODEL_ID, **agent_model_kwargs),
    instructions=[
        "Answer ONLY 'YES' if the DB likely contains relevant documents for the query, otherwise answer 'NO'.",
    ],
    markdown=False,
)

db_agent = Agent(
    name="DBAccessAgent",
    model=OpenAIChat(id=MODEL_ID, **agent_model_kwargs),
    instructions=["You are given document chunks. Provide a concise, relevant answer for the user's query."],
    markdown=True,
)

def runresponse_to_text(run_resp) -> str:
    if run_resp is None:
        return ""
    content = getattr(run_resp, "content", None)
    if content:
        if isinstance(content, str):
            return content.strip()
        try:
            return json.dumps(content, indent=2, ensure_ascii=False)
        except Exception:
            return str(content)
    messages = getattr(run_resp, "messages", None)
    if messages:
        try:
            parts = []
            for m in messages:
                c = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
                if c:
                    parts.append(c)
            if parts:
                return "\n".join(parts).strip()
        except Exception:
            return str(messages)
    return str(run_resp)

def check_metadata_with_vector(query: str, threshold_hits: int = 1) -> bool:
    try:
        print("Vector similarity_search starting...")
        hits = vectorstore.similarity_search(query, k=3)
        print("Vector search done. hit count:", len(hits))
        return len(hits) >= threshold_hits
    except Exception as e:
        print("Vectorstore check failed:", type(e).__name__, e)
        return False

def get_top_chunks(query: str, k: int = 3):
    print("Retrieving top chunks from vectorstore...")
    return vectorstore.similarity_search(query, k=k)

def trim_chunks(chunks, max_chars=2500):
    """Concatenate but trim to max_chars total (safe smaller prompt)."""
    out = []
    total = 0
    for c in chunks:
        s = c.page_content.strip()
        if total + len(s) > max_chars:
            # keep a portion
            remaining = max_chars - total
            if remaining <= 0:
                break
            s = s[:remaining].rsplit("\n", 1)[0]  # avoid cutting mid-line
        out.append(s)
        total += len(s)
        if total >= max_chars:
            break
    return out

def main():
    user_query = input("Enter your query: ").strip()
    if not user_query:
        print("Empty query. Exiting.")
        return

    # Ask metadata agent (LLM) with timeout
    try:
        print("Calling metadata_agent.run() ...")
        meta_run = metadata_agent.run(user_query)
        meta_text = runresponse_to_text(meta_run)
        print("metadata_agent returned:", textwrap.shorten(meta_text, width=200))
    except Exception as e:
        print("Metadata agent failed:", type(e).__name__, e)
        meta_text = ""

    decision_text = (meta_text or "").strip().upper()
    # deterministic vector check
    found = check_metadata_with_vector(user_query)

    print("Decision from LLM:", decision_text, "| Found by vector:", found)

    if decision_text == "YES" or found:
        try:
            chunks = get_top_chunks(user_query, k=5)
            # Trim each chunk and overall payload to protect the model call
            trimmed = trim_chunks(chunks, max_chars=3000)
            print(f"Using {len(trimmed)} trimmed chunks (total chars <= 3000).")
            content = "\n\n---\n\n".join(trimmed)

            db_input = f"User query: {user_query}\n\nRetrieved chunks:\n{content}\n\nReturn the most relevant chunks concisely."
            print("Calling db_agent.run() with trimmed content (will timeout after 30s).")
            db_run = db_agent.run(db_input)
            db_text = runresponse_to_text(db_run)
            print("\n===== ANSWER =====\n")
            print(db_text)
        except Exception as e:
            print("Error retrieving or processing chunks:", type(e).__name__, e)
            traceback.print_exc()
    else:
        print("No relevant documents found in the DB.")

if __name__ == "__main__":
    main()

'''
output:
===== ANSWER =====

Your query relates to happiness, and one relevant excerpt is:

- "My own complete happiness, and the home-centred..." 

This indicates a personal focus on achieving happiness related to one's own life and relationships.

'''    
