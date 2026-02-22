import hashlib
import json
import os
import re
import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = "./index_store"
DOCS_JSONL = os.path.join(INDEX_DIR, "docs.jsonl")
DB_PATH = "./chroma_db"
BATCH_SIZE = 100  # lower this if you run out of RAM


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _write_jsonl(path: str, rows: List[dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sanitize_metadata(meta: dict) -> dict:
    # PyMuPDF sometimes returns weird types (Rect, Matrix, etc.) that chroma hates
    return {
        k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
        for k, v in meta.items()
    }


def _make_chunk_id(pdf_file: str, page: int, start_index: int) -> str:
    # stable id so re-ingesting the same file doesn't create duplicates
    key = f"{pdf_file}::{page}::{start_index}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _natural_sort_key(name: str) -> list:
    # makes doc_9.pdf sort before doc_10.pdf
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def ingest_all_pdfs(folder_path: str = "data") -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"created '{folder_path}', drop your PDFs in there and run again")
        return

    pdf_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")],
        key=_natural_sort_key,
    )
    if not pdf_files:
        print("no PDFs found, nothing to do")
        return

    # wipe old data so we don't end up with stale chunks mixed in
    for path, label in [(DB_PATH, "chroma db"), (INDEX_DIR, "index store")]:
        if os.path.isdir(path):
            print(f"removing old {label}...")
            shutil.rmtree(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    jsonl_rows = []
    skipped: List[str] = []

    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        print(f"loading: {pdf_file}")

        try:
            pages = PyMuPDFLoader(file_path).load()
        except Exception as exc:
            print(f"  couldn't load {pdf_file}, skipping ({exc})")
            skipped.append(pdf_file)
            continue

        for doc in pages:
            doc.metadata["source_file"] = pdf_file
            doc.metadata.setdefault("page", None)

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            page = chunk.metadata.get("page") or 0
            start = chunk.metadata.get("start_index") or 0
            chunk.metadata["chunk_id"] = _make_chunk_id(pdf_file, page, start)
            chunk.metadata = _sanitize_metadata(chunk.metadata)

            jsonl_rows.append({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            })
            all_chunks.append(chunk)

    if not all_chunks:
        print("no chunks produced,something probably went wrong above")
        return

    print(f"total chunks: {len(all_chunks)}")

    print("writing jsonl store...")
    _write_jsonl(DOCS_JSONL, jsonl_rows)

    print("building embeddings (nomic-embed-text via ollama)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)

    total_batches = -(-len(all_chunks) // BATCH_SIZE)
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        db.add_documents(batch)
        print(f"  batch {i // BATCH_SIZE + 1}/{total_batches}")

    print("\ndone.")
    print(f"  chroma db  -> {DB_PATH}")
    print(f"  jsonl      -> {DOCS_JSONL}")
    print(f"  chunks     -> {len(all_chunks)}")
    if skipped:
        print(f"  skipped    -> {', '.join(skipped)}")


if __name__ == "__main__":
    ingest_all_pdfs("data")
