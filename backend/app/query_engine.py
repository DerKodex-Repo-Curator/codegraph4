import os
from git import Repo
from app.code_loader import load_repo_files
from app.chunking import chunk_file_list
from app.vector_store import index_chunks, collection
from app.retrieval import hybrid_retrieval
import requests


def is_repo_indexed(repo_name):
    """
    Checks whether this repo already has chunks stored in Chroma,
    so we don't waste time re-indexing on every single query.
    """
    results = collection.get(where={"repo_name": repo_name}, limit=1)
    return len(results["ids"]) > 0


def clone_repo(repo_url):
    """
    Clones the repo if not already present locally.
    Returns (repo_name, local_path).
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_path = f"repos/{repo_name}"

    if not os.path.exists("repos"):
        os.makedirs("repos")

    if not os.path.exists(clone_path):
        Repo.clone_from(repo_url, clone_path)

    return repo_name, clone_path

def answer_query(repo_url, query):
    """
    The main entry point: given a repo URL and a question, returns an answer
    grounded in the most relevant retrieved code chunks.
    """
    repo_name, clone_path = clone_repo(repo_url)

    # Only chunk + index if we haven't done this repo before —
    # this is the "fast path vs slow path" distinction from earlier
    if not is_repo_indexed(repo_name):
        print(f"'{repo_name}' not indexed yet — chunking and indexing now...")
        files = load_repo_files(clone_path)
        chunks = chunk_file_list(files)
        index_chunks(chunks, repo_name=repo_name)
    else:
        print(f"'{repo_name}' already indexed — skipping straight to retrieval.")
        # We still need the chunk list in memory for BM25 (which isn't
        # stored in Chroma) — so we re-read and re-chunk the files here.
        # This is fast (no embedding calls), just file reading + AST parsing.
        files = load_repo_files(clone_path)
        chunks = chunk_file_list(files)

    # Retrieve the top 5 most relevant chunks for this specific query
    top_chunks = hybrid_retrieval(query, chunks, top_k=5, repo_name=repo_name)

    # Build a prompt using ONLY these 5 chunks, not the whole repo
    context = ""
    for c in top_chunks:
        label = f"{c['file_path']} :: {c['symbol_name']}" if c['symbol_name'] else c['file_path']
        context += f"\n\nFILE: {label}\n{c['code']}"

    answer = ask_ollama(query, context)
    return answer

def ask_ollama(query, context):
    prompt = f"""You are a senior software engineer helping someone understand a codebase.

Answer the following question using ONLY the code provided below. Be clear and specific.

Question: {query}

Relevant code:
{context}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code != 200:
            return f"Error: {response.text}"

        data = response.json()
        return data.get("response", "No response from model")

    except Exception as e:
        return f"Ollama error: {str(e)}"