import os
import json
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

DB_PATH = "./chroma_db"
DOCS_JSONL = "./index_store/docs.jsonl"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _display_page(page) -> str:
    return str(int(page) + 1) if isinstance(page, int) else "N/A"


@st.cache_data(show_spinner="Loading chunk store...")
def load_jsonl_docs(path: str) -> List[dict]:
    docs = []
    if not os.path.exists(path):
        return docs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    errors = []
    if not os.path.isdir(DB_PATH):
        errors.append(f"Chroma DB not found at `{DB_PATH}`.")
    if not os.path.exists(DOCS_JSONL):
        errors.append(f"Chunk store not found at `{DOCS_JSONL}`.")
    if errors:
        return None, None, errors

    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vs = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        llm = ChatOllama(model="llama3.2", temperature=0)
        return vs, llm, []
    except Exception as e:
        return None, None, [str(e)]


@st.cache_resource(show_spinner="Building BM25 index...")
def load_bm25(_docs_rows: List[dict]):
    # leading underscore so streamlit doesn't try to hash this arg
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        documents = [
            Document(page_content=r["page_content"], metadata=r.get("metadata", {}))
            for r in _docs_rows
        ]
        return BM25Retriever.from_documents(documents)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading reranker (first run downloads ~80 MB)...")
def load_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL)
    except ImportError:
        return None
    except Exception:
        return None


def bm25_rank(bm25_retriever, docs_rows: List[dict], query: str, k: int = 5) -> List[dict]:
    if bm25_retriever is not None:
        try:
            bm25_retriever.k = k
            results = bm25_retriever.invoke(query)
            return [
                {"page_content": d.page_content, "metadata": d.metadata, "_kw_score": 1.0}
                for d in results
            ]
        except Exception:
            pass

    # fallback: basic token overlap if BM25 isn't available
    q_tokens = query.lower().split()
    scored = []
    for r in docs_rows:
        score = sum(1 for tok in q_tokens if tok in r["page_content"].lower())
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**r, "_kw_score": float(s)} for s, r in scored[:k]]


def semantic_search(vs: Chroma, query: str, k: int, mode: str) -> List[dict]:
    try:
        if mode == "mmr":
            results = vs.max_marginal_relevance_search(query, k=k, fetch_k=20)
            return [
                {"page_content": d.page_content, "metadata": d.metadata, "_sem_score": None}
                for d in results
            ]
        else:
            results = vs.similarity_search_with_relevance_scores(query, k=k)
            return [
                {"page_content": d.page_content, "metadata": d.metadata, "_sem_score": float(s)}
                for d, s in results
            ]
    except Exception:
        docs = vs.similarity_search(query, k=k)
        return [
            {"page_content": d.page_content, "metadata": d.metadata, "_sem_score": None}
            for d in docs
        ]


def dedupe_docs(rows: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in rows:
        meta = r.get("metadata", {})
        key = (meta.get("source_file"), meta.get("page"), meta.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def apply_source_filter(rows: List[dict], source_file: str) -> List[dict]:
    if source_file == "All":
        return rows
    return [r for r in rows if r.get("metadata", {}).get("source_file") == source_file]


def rerank(query: str, rows: List[dict], reranker, top_n: int) -> List[dict]:
    pairs = [(query, r["page_content"]) for r in rows]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, rows), key=lambda x: float(x[0]), reverse=True)
    return [{**r, "_rerank_score": round(float(s), 4)} for s, r in ranked[:top_n]]


def rewrite_query(llm, history: list, current_query: str) -> str:
    # no history means nothing to rewrite against
    if len(history) < 2:
        return current_query
    prompt = (
        "Given the conversation history below, rewrite the final question as a "
        "self-contained search query. Output only the rewritten query, nothing else.\n\n"
        f"History: {history}\n"
        f"Final question: {current_query}"
    )
    try:
        result = llm.invoke(prompt)
        rewritten = result.content.strip()
        return rewritten if rewritten else current_query
    except Exception:
        return current_query


def retrieve_hybrid(
    vs: Chroma,
    docs_rows: List[dict],
    bm25_retriever,
    query: str,
    k_sem: int,
    k_kw: int,
    k_final: int,
    retrieval_mode: str,
    reranker,
    rerank_enabled: bool,
    source_choice: str,
) -> Tuple[List[dict], Dict[str, Any]]:
    sem_rows = apply_source_filter(
        semantic_search(vs, query, k=k_sem, mode=retrieval_mode), source_choice
    )
    kw_rows = (
        bm25_rank(bm25_retriever, apply_source_filter(docs_rows, source_choice), query, k=k_kw)
        if k_kw > 0 and docs_rows
        else []
    )

    combined = dedupe_docs(sem_rows + kw_rows)

    if rerank_enabled and reranker is not None and combined:
        final_rows = rerank(query, combined, reranker, top_n=k_final)
    else:
        final_rows = combined[:k_final]

    best_sem: Optional[float] = None
    for r in sem_rows:
        s = r.get("_sem_score")
        if s is not None:
            best_sem = s if best_sem is None else max(best_sem, s)

    debug_info = {
        "best_semantic_relevance": best_sem,
        "semantic_count": len(sem_rows),
        "keyword_count": len(kw_rows),
        "final_count": len(final_rows),
        "reranked": rerank_enabled and reranker is not None,
        "effective_query": query,
    }
    return final_rows, debug_info


def should_refuse(
    debug_info: Dict[str, Any],
    final_rows: List[dict],
    retrieval_mode: str,
    threshold: float,
) -> bool:
    if not final_rows:
        return True

    sem_hits = debug_info.get("semantic_count", 0)
    kw_hits = debug_info.get("keyword_count", 0)

    if sem_hits == 0 and kw_hits == 0:
        return True

    # only bail on low confidence if keyword search also came up empty
    if retrieval_mode == "similarity":
        best = debug_info.get("best_semantic_relevance")
        if best is not None and best < threshold and kw_hits == 0:
            return True

    return False


def format_context(docs: List[dict], max_chars_each: int = 1600) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        source = meta.get("source_file", "Unknown")
        page_str = _display_page(meta.get("page"))
        content = d.get("page_content", "")
        if len(content) > max_chars_each:
            content = content[:max_chars_each] + "… [truncated]"
        parts.append(f"[{i}] Source: {source} | Page: {page_str}\n{content}")
    return "\n\n".join(parts)


# UI

st.set_page_config(page_title="Private Technical Assistant", page_icon="🛡️", layout="wide")
st.title("Private Technical Assistant")
st.caption("Fully local RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

vs, llm, load_errors = load_resources()

if load_errors:
    st.error("Could not load required resources. Please run `ingest.py` first.")
    for err in load_errors:
        st.caption(f"• {err}")
    st.stop()

docs_rows = load_jsonl_docs(DOCS_JSONL)
bm25_retriever = load_bm25(tuple(r["page_content"] for r in docs_rows))

with st.sidebar:
    st.header("Retrieval Settings")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    debug = st.toggle("Debug mode", value=False)
    retrieval_mode = st.selectbox("Semantic search type", ["similarity", "mmr"], index=0)
    k_sem = st.slider("Semantic k", 1, 12, 6)
    k_kw = st.slider("Keyword k (BM25)", 0, 12, 4)
    k_final = st.slider("Final chunks to LLM", 2, 10, 5)
    threshold = st.slider("Guardrail threshold (semantic relevance)", 0.0, 1.0, 0.20, 0.01)
    st.caption("If confidence is low, the assistant will say the manual doesn't contain the info.")

    query_rewrite = st.toggle("Query rewriting (multi-turn)", value=True)
    st.caption(
        "Rewrites follow-up questions as standalone queries before retrieval, "
        "so context is correctly grounded for every turn."
    )

    st.divider()
    st.subheader("Reranking")
    rerank_enabled = st.toggle("Enable reranking", value=False)
    st.caption(
        f"Uses `{RERANKER_MODEL}` (~80 MB, downloads once on first enable). "
        "Rescores retrieved chunks with a cross-encoder for better relevance ordering."
    )

    reranker = None
    if rerank_enabled:
        reranker = load_reranker()
        if reranker is None:
            st.warning("Reranker failed to load. Run: `pip install sentence-transformers`")
        else:
            st.success("Reranker active")

    st.divider()
    st.header("Source Filter")
    available_sources = sorted(
        {r.get("metadata", {}).get("source_file", "Unknown") for r in docs_rows}
    )
    source_choice = st.selectbox("Use only this PDF", ["All"] + available_sources, index=0)


SYSTEM_PROMPT = """You are a technical documentation expert.

Rules:
- Use ONLY the provided context.
- If the context does not contain the answer, say: "The manual does not contain this information."
- Be precise, structured, and step-by-step when procedures are requested.
- When you use a fact, cite it using [#] from the context.

Context:
{context}"""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about the manual..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        refused = False

        with st.spinner("Retrieving context..."):
            effective_query = (
                rewrite_query(llm, st.session_state.messages[:-1], query)
                if query_rewrite
                else query
            )

            rows, debug_info = retrieve_hybrid(
                vs=vs,
                docs_rows=docs_rows,
                bm25_retriever=bm25_retriever,
                query=effective_query,
                k_sem=k_sem,
                k_kw=k_kw,
                k_final=k_final,
                retrieval_mode=retrieval_mode,
                reranker=reranker,
                rerank_enabled=rerank_enabled,
                source_choice=source_choice,
            )

        if should_refuse(debug_info, rows, retrieval_mode, threshold):
            full_response = "The manual does not contain this information."
            refused = True
            response_placeholder.markdown(full_response)

        else:
            context = format_context(rows)

            if debug:
                with st.expander("Debug: Context preview"):
                    st.code(context[:6000])
                with st.expander("Debug: Retrieval stats"):
                    st.json(debug_info)

            lc_messages = [("system", SYSTEM_PROMPT.format(context=context))]
            for m in st.session_state.messages[:-1]:
                role = "human" if m["role"] == "user" else "assistant"
                lc_messages.append((role, m["content"]))
            lc_messages.append(("human", query))

            for chunk in llm.stream(lc_messages):
                token = getattr(chunk, "content", None)
                if token:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        if not refused:
            with st.expander("View Source Chunks"):
                if not rows:
                    st.info("No source chunks retrieved.")
                else:
                    for i, r in enumerate(rows, 1):
                        meta = r.get("metadata", {})
                        src = meta.get("source_file", "Unknown")
                        page_str = _display_page(meta.get("page"))
                        content = r.get("page_content", "")

                        st.markdown(f"**[{i}] {src} | Page {page_str}**")
                        if debug:
                            st.caption(
                                f"sem={r.get('_sem_score')} | "
                                f"kw={r.get('_kw_score')} | "
                                f"rerank={r.get('_rerank_score')}"
                            )
                        st.info(content[:2000] + ("…" if len(content) > 2000 else ""))

        st.session_state.messages.append({"role": "assistant", "content": full_response})
