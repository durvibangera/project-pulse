"""
src/nlp/summarizer.py
----------------------
Evidence Summarizer — LangChain RetrievalQA chain using local Mistral via Ollama.

Architecture:
    User query (condition + drug)
        ↓
    ChromaDB similarity search → top-K review chunks
        ↓
    LangChain RetrievalQA  →  Mistral 7B (Ollama, local)
        ↓
    Structured clinical summary:
        • Effectiveness  • Side effects  • Sentiment  • Recommendation

Requirements:
    - Ollama running locally: https://ollama.com/
    - Model pulled: ollama pull mistral
    - ChromaDB index built: python src/nlp/rag_pipeline.py

Usage:
    python src/nlp/summarizer.py

    # Programmatic
    from src.nlp.summarizer import summarize_drug
    result = summarize_drug(condition="Type 2 Diabetes", drug="Metformin")
    print(result["summary"])
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.nlp.rag_pipeline import load_drug_index

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OLLAMA_MODEL = "mistral"   # pulled via: ollama pull mistral
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical evidence summarizer for a Global Data Science & Analytics team.
You have been given a set of patient drug reviews from a medical database.
Produce a concise, structured evidence summary using ONLY the information in the reviews below.

Patient Reviews:
{context}

Request: {question}

Respond in this exact format:
**Drug Effectiveness:** [1-2 sentences on how effective patients found it]
**Common Side Effects:** [bullet list of most frequently mentioned side effects]
**Overall Patient Sentiment:** [Positive / Mixed / Negative — with brief justification]
**Clinical Recommendation:** [1 sentence recommendation for the clinical team]
""",
)

# ------------------------------------------------------------------
# Core summarizer
# ------------------------------------------------------------------

def summarize_drug(
    condition: str,
    drug: str | None = None,
    top_k: int = 8,
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Retrieve relevant drug reviews and generate a clinical summary.

    Parameters
    ----------
    condition : str
        Clinical condition to query, e.g. "Type 2 Diabetes", "depression"
    drug : str, optional
        Specific drug name to focus on, e.g. "Metformin"
    top_k : int
        Number of review chunks to retrieve from ChromaDB
    model : str
        Ollama model name (default: "mistral")

    Returns
    -------
    dict with keys:
        "drug"        : queried drug name (or "any")
        "condition"   : queried condition
        "top_k"       : chunks retrieved
        "summary"     : full LLM-generated summary string
        "source_docs" : list of retrieved review snippets
    """
    print(f"[NLP] Loading ChromaDB index ...")
    vectorstore = load_drug_index()

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    question = (
        f"Summarize patient experiences with {drug} for {condition}."
        if drug
        else f"Summarize patient experiences with medications for {condition}."
    )

    print(f"[NLP] Retrieving source documents ...")
    source_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in source_docs)

    print(f"[NLP] Initialising Ollama/{model} ...")
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    print(f"[NLP] Generating summary (Mistral 7B) ...")
    prompt_val = _SYSTEM_PROMPT.format(context=context_text, question=question)
    summary = (llm.invoke(prompt_val)).content

    # Pretty-print to console
    _print_summary(drug or "any", condition, summary, source_docs)

    return {
        "drug":        drug or "any",
        "condition":   condition,
        "top_k":       top_k,
        "summary":     summary,
        "source_docs": [
            {
                "drug":      d.metadata.get("drug"),
                "condition": d.metadata.get("condition"),
                "rating":    d.metadata.get("rating"),
                "snippet":   d.page_content[:200],
            }
            for d in source_docs
        ],
    }


# ------------------------------------------------------------------
# Fallback: pure retrieval summary (no LLM — for when Ollama is down)
# ------------------------------------------------------------------

def summarize_drug_no_llm(
    condition: str,
    drug: str | None = None,
    top_k: int = 10,
) -> dict:
    """
    Lightweight fallback — returns retrieved review snippets without LLM synthesis.
    Useful for dashboard display when Ollama is unavailable.
    """
    vectorstore = load_drug_index()
    query = f"{drug or ''} {condition} benefits side effects".strip()
    docs  = vectorstore.similarity_search(query, k=top_k)

    ratings   = [d.metadata.get("rating", 0) for d in docs]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
    sentiment = (
        "Positive" if avg_rating >= 7
        else "Mixed"   if avg_rating >= 4
        else "Negative"
    )

    snippets = [
        f"• [{d.metadata.get('drug')} / {d.metadata.get('condition')} "
        f"rating={d.metadata.get('rating'):.0f}/10] "
        f"{d.page_content[:150].replace(chr(10),' ')}"
        for d in docs
    ]

    summary = (
        f"[Retrieval-only summary — Ollama unavailable]\n\n"
        f"Top {top_k} matching reviews for '{drug or 'medications'}' in '{condition}':\n"
        f"Average rating: {avg_rating:.1f}/10 → Overall sentiment: {sentiment}\n\n"
        + "\n".join(snippets)
    )

    return {
        "drug":        drug or "any",
        "condition":   condition,
        "top_k":       top_k,
        "summary":     summary,
        "source_docs": [
            {
                "drug":      d.metadata.get("drug"),
                "condition": d.metadata.get("condition"),
                "rating":    d.metadata.get("rating"),
                "snippet":   d.page_content[:200],
            }
            for d in docs
        ],
    }


# ------------------------------------------------------------------
# Console renderer
# ------------------------------------------------------------------

def _print_summary(drug: str, condition: str, summary: str, source_docs) -> None:
    print()
    print("=" * 64)
    print(f"  CLINICAL EVIDENCE SUMMARY")
    print(f"  Drug: {drug}  |  Condition: {condition}")
    print("=" * 64)
    print(summary)
    print()
    print(f"  Sources: {len(source_docs)} review chunks retrieved")
    for d in source_docs[:3]:
        m = d.metadata
        print(f"    • {m.get('drug')} ({m.get('condition')}) rating={m.get('rating'):.0f}/10")
    print("=" * 64 + "\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P.U.L.S.E. Drug Evidence Summarizer")
    parser.add_argument("--condition", default="Type 2 Diabetes",
                        help="Clinical condition to query")
    parser.add_argument("--drug",      default="Metformin",
                        help="Drug name (optional)")
    parser.add_argument("--no-llm",    action="store_true",
                        help="Use retrieval-only fallback (no Ollama required)")
    args = parser.parse_args()

    if args.no_llm:
        result = summarize_drug_no_llm(condition=args.condition, drug=args.drug)
    else:
        result = summarize_drug(condition=args.condition, drug=args.drug)

    print(f"\n[NLP] Done. Summary length: {len(result['summary'])} chars")
