"""
src/nlp/rag_pipeline.py
------------------------
Evidence Summarizer — ChromaDB index builder for drug reviews.

Indexes all ~4,100 drug reviews (train + test) from Druglib.com into a
persistent ChromaDB vector store using free, local HuggingFace embeddings
(all-MiniLM-L6-v2). No API key required.

Each review is stored as a chunked Document with metadata:
    drug, condition, rating, effectiveness, side_effects

Usage:
    # Build the index (first run — downloads ~90MB embedding model)
    python src/nlp/rag_pipeline.py

    # Programmatic access from other modules
    from src.nlp.rag_pipeline import build_drug_index, load_drug_index
    vectorstore = load_drug_index()
    docs = vectorstore.similarity_search("metformin type 2 diabetes", k=5)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # ~90MB, fully local
CHROMA_DIR  = "./chroma_db"

_TRAIN_PATH = "data/raw/drug-reviews/drugLibTrain_raw.tsv"
_TEST_PATH  = "data/raw/drug-reviews/drugLibTest_raw.tsv"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_reviews() -> pd.DataFrame:
    """Load and concatenate both TSV splits."""
    train = pd.read_csv(_TRAIN_PATH, sep="\t", encoding="latin-1")
    test  = pd.read_csv(_TEST_PATH,  sep="\t", encoding="latin-1")
    df = pd.concat([train, test], ignore_index=True)
    # drop the stray unnamed index column
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


def _build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert each review row into a LangChain Document.
    The page_content concatenates all free-text fields for rich semantic search.
    Structured fields are stored as metadata for post-retrieval filtering.
    """
    docs = []
    for _, row in df.iterrows():
        drug      = str(row.get("urlDrugName", "Unknown")).strip()
        condition = str(row.get("condition",   "Unknown")).strip()
        rating    = row.get("rating", "")
        effectiv  = str(row.get("effectiveness",     "")).strip()
        side_eff  = str(row.get("sideEffects",       "")).strip()
        benefits  = str(row.get("benefitsReview",    "")).strip()
        side_rev  = str(row.get("sideEffectsReview", "")).strip()
        comments  = str(row.get("commentsReview",    "")).strip()

        page_content = (
            f"Drug: {drug}\n"
            f"Condition: {condition}\n"
            f"Effectiveness: {effectiv}\n"
            f"Benefits: {benefits}\n"
            f"Side Effects (label): {side_eff}\n"
            f"Side Effects (review): {side_rev}\n"
            f"Comments: {comments}\n"
            f"Rating: {rating}/10"
        )

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "drug":         drug,
                    "condition":    condition,
                    "rating":       float(rating) if rating != "" else 0.0,
                    "effectiveness":effectiv,
                    "side_effects": side_eff,
                },
            )
        )
    return docs


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_drug_index(force_rebuild: bool = False) -> Chroma:
    """
    Build (or rebuild) the ChromaDB vector store from all drug reviews.

    Parameters
    ----------
    force_rebuild : bool
        If True, delete the existing chroma_db and rebuild from scratch.
        Defaults to False — skips rebuild if the store already exists.

    Returns
    -------
    Chroma vectorstore instance ready for similarity_search() calls.
    """
    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        print(f"[NLP] ChromaDB already exists at {CHROMA_DIR} — loading existing index.")
        print("[NLP] Pass force_rebuild=True to rebuild from scratch.")
        return load_drug_index()

    if force_rebuild and os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"[NLP] Removed existing ChromaDB at {CHROMA_DIR}")

    print("[NLP] Loading drug reviews ...")
    df = _load_reviews()
    print(f"[NLP] Total reviews : {len(df):,} | Drugs : {df.get('urlDrugName', pd.Series()).nunique()}")

    print("[NLP] Building documents ...")
    docs = _build_documents(df)

    print("[NLP] Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[NLP] {len(chunks):,} chunks ready for indexing.")

    print(f"[NLP] Loading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[NLP] Indexing into ChromaDB (this may take 1-3 minutes) ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Count persisted docs
    n = vectorstore._collection.count()
    print(f"[NLP] Index complete — {n:,} vectors stored in {CHROMA_DIR}")
    return vectorstore


def load_drug_index() -> Chroma:
    """
    Load an existing ChromaDB index.  Call build_drug_index() first if it
    does not exist.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Run `python src/nlp/rag_pipeline.py` to build the index first."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    vs = build_drug_index()
    # Quick smoke-test
    print("\n[NLP] Smoke test — searching for 'metformin type 2 diabetes' ...")
    hits = vs.similarity_search("metformin type 2 diabetes benefits side effects", k=3)
    for i, doc in enumerate(hits, 1):
        meta = doc.metadata
        print(f"  [{i}] {meta.get('drug')} | {meta.get('condition')} | rating={meta.get('rating')}")
        print(f"       {doc.page_content[:120].replace(chr(10), ' ')}")
