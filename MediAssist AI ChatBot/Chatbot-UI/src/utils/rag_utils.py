"""
RAG (Retrieval-Augmented Generation) utilities for document processing and similarity search.
"""

import io
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional PDF parsing
try:
    import pypdf
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False


def read_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    if not HAVE_PYPDF:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n\n".join(texts)
    except Exception:
        return ""


def split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for RAG processing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur: 
                out.append(cur)
            cur = p
    if cur: 
        out.append(cur)
    
    # Add simple overlap
    if overlap > 0 and len(out) > 1:
        new = []
        for i, c in enumerate(out):
            if i == 0: 
                new.append(c)
            else:
                prev_tail = out[i-1][-overlap:]
                new.append((prev_tail + " " + c).strip())
        out = new
    return out


def make_rag_index(docs: List[Tuple[str, str]], max_features: int = 40000) -> Optional[Dict]:
    """
    Create a TF-IDF based RAG index from documents.
    
    Args:
        docs: List of (filename, text) tuples
        max_features: Maximum number of features for TF-IDF vectorizer
        
    Returns:
        Dictionary containing vectorizer, matrix, metadata, and corpus, or None if no documents
    """
    corpus, meta = [], []
    for name, text in docs:
        for i, ch in enumerate(split_into_chunks(text)):
            corpus.append(ch)
            meta.append({"source": name, "chunk_id": i})
    
    if not corpus:
        return None
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return {
        "vectorizer": vectorizer, 
        "X": X, 
        "meta": meta, 
        "corpus": corpus
    }


def rag_retrieve(index: Optional[Dict], query: str, k: int = 4) -> List[Dict]:
    """
    Retrieve top-k most similar chunks for a query.
    
    Args:
        index: RAG index dictionary
        query: Search query
        k: Number of top results to return
        
    Returns:
        List of dictionaries with text, source, score, and chunk_id
    """
    if not index: 
        return []
    
    try:
        qv = index["vectorizer"].transform([query])
        sims = cosine_similarity(qv, index["X"]).ravel()
        topk = sims.argsort()[::-1][:k]
        results = []
        for i in topk:
            results.append({
                "text": index["corpus"][i],
                "source": index["meta"][i]["source"],
                "score": float(sims[i]),
                "chunk_id": int(index["meta"][i]["chunk_id"]),
            })
        return results
    except Exception:
        return []


def format_context(snippets: List[Dict]) -> str:
    """
    Format retrieved snippets into a context string for the LLM.
    
    Args:
        snippets: List of retrieved snippet dictionaries
        
    Returns:
        Formatted context string
    """
    if not snippets: 
        return ""
    
    lines = []
    for s in snippets:
        head = f"[{s['source']} • chunk {s['chunk_id']} • score={s['score']:.3f}]"
        body = s["text"].strip().replace("\n", " ")
        lines.append(f"{head}\n{body}")
    return "\n\n".join(lines)
