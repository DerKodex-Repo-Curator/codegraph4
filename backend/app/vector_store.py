import chromadb
from app.embeddings import get_embedding

# persistantclient tell the chroma to save the data in this folder to disk, so we can reuse it later
client = chromadb.PersistentClient(path="./chroma_data")

# collection (table)

collection = client.get_or_create_collection( name = "code_chunks") # it will check if the collection exists, if not it will create it

# Indexing phase: here we will take the chunks from chunk_file_list and add them .
# this will only be done once, and then we can reuse the collection for retrieval

def index_chunks(chunks, repo_name):
    """
    INDEXING PHASE. 'repo_name' tags every chunk so we can later filter
    searches to only look within this specific repo — since multiple
    repos will eventually share the same Chroma collection.
    """
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for c in chunks:
        # Prefix the chunk ID with repo_name too — so the SAME file/symbol
        # name in two different repos doesn't collide (e.g. two repos
        # could both have a "main.py::home" function)
        chunk_id = f"{repo_name}::{c['file_path']}::{c['symbol_name']}"

        embedding = get_embedding(c["code"])

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(c["code"])
        metadatas.append({
            "repo_name": repo_name,
            "file_path": c["file_path"],
            "symbol_name": c["symbol_name"] or "",
            "type": c["type"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Indexed {len(chunks)} chunks for repo '{repo_name}' into Chroma.")
# this querying phase where this runs everytime user make a query, and it will return the top_k most relevant chunks based on the query embedding
def vector_retrieval(query, chunks=None, top_k=5, repo_name=None):
    """
    QUERYING PHASE. If repo_name is given, only searches within that repo's
    chunks — otherwise searches across everything stored in Chroma.
    """
    query_embedding = get_embedding(query)

    # 'where' is Chroma's filtering mechanism — it only considers chunks
    # whose metadata matches this condition, before ranking by similarity
    where_filter = {"repo_name": repo_name} if repo_name else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "file_path": results["metadatas"][0][i]["file_path"],
            "symbol_name": results["metadatas"][0][i]["symbol_name"] or None,
            "code": results["documents"][0][i]
        })

    return retrieved