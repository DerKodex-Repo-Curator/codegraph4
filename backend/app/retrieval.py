from rank_bm25 import BM25Okapi
from app.vector_store import vector_retrieval



def tokenize(text):
    """
    Simple tokenizer: lowercase, split on whitespace, strip common code punctuation.
    Good enough for BM25 — it just needs words, not perfect NLP tokenization.
    """
    import re
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    return tokens


def build_bm25_index(chunks):
    """
    Builds a BM25 index over a list of chunks.
    Returns the index plus the chunk list (so we can map results back).
    """
    # tokenize each chunk's code (this is what BM25 searches over)
    tokenized_chunks = [tokenize(c["code"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


def bm25_retrieval(query, chunks, top_k=5):
    """
    Matches the same interface as baseline_retrieval:
    takes a query + full chunk list, returns top_k ranked chunks.
    """
    bm25 = build_bm25_index(chunks)
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    # pair each chunk with its score, sort descending, take top_k
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [c for c, score in scored_chunks[:top_k]]


from app.embeddings import get_embedding
import math


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)


def embedding_retrieval(query, chunks, top_k=5):
    """
    Same interface as baseline_retrieval / bm25_retrieval.
    Now uses Chroma's pre-indexed embeddings instead of live-embedding
    every chunk on every call — fast and consistent.
    """
    return vector_retrieval(query, top_k=top_k)

def normalize_scores(scored_list):
    """
    Takes a list of (chunk, score) tuples and rescales scores to 0-1
    so BM25 and embedding scores become comparable.
    """
    scores = [s for _, s in scored_list]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(c, 1.0) for c, s in scored_list]
    return [(c, (s - min_s) / (max_s - min_s)) for c, s in scored_list]


def hybrid_retrieval(query, chunks, top_k=5, bm25_weight=0.3, repo_name=None):
    """
    Combines BM25 and embedding scores into one ranked list.
    repo_name filters the embedding search to only this repo's chunks
    (BM25 already only sees whatever's in 'chunks', so no change needed there).
    """
    # --- BM25 scoring (unchanged) ---
    bm25 = build_bm25_index(chunks)
    tokenized_query = tokenize(query)
    bm25_raw_scores = bm25.get_scores(tokenized_query)
    bm25_scored = list(zip(chunks, bm25_raw_scores))
    bm25_normalized = normalize_scores(bm25_scored)
    bm25_scores_by_id = {chunk_id(c): s for c, s in bm25_normalized}

    # --- Embedding scoring via Chroma, now filtered to this repo ---
    vector_candidates = vector_retrieval(query, top_k=20, repo_name=repo_name)
    embed_scores_by_id = {}
    for rank, c in enumerate(vector_candidates):
        cid = chunk_id(c)
        embed_scores_by_id[cid] = 1 - (rank / len(vector_candidates))

    # --- Combine (unchanged) ---
    combined = []
    for c in chunks:
        cid = chunk_id(c)
        bm25_score = bm25_scores_by_id.get(cid, 0)
        embed_score = embed_scores_by_id.get(cid, 0)
        final_score = (bm25_weight * bm25_score) + ((1 - bm25_weight) * embed_score)
        combined.append((c, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [c for c, score in combined[:top_k]]
def chunk_id(chunk):
    return f"{chunk['file_path']}::{chunk['symbol_name']}"