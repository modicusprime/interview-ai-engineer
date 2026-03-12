"""
FDA Warning Letters — RAG Question Answering (Terminal)
========================================================
Uses:
  - HuggingFace Inference API (free) for the LLM
  - sentence-transformers for embeddings (runs locally, free)
  - FAISS for the vector store (local, no server)

Install dependencies:
    pip install langchain langchain-community langchain-huggingface
    pip install faiss-cpu sentence-transformers huggingface_hub

Setup:
    1. Create a free account at https://huggingface.co
    2. Go to https://huggingface.co/settings/tokens and create a token
    3. Set it as an environment variable:
         Windows:  set HF_TOKEN=hf_your_token_here
         Mac/Linux: export HF_TOKEN=hf_your_token_here
       Or paste it directly into HF_TOKEN below (not recommended for shared code)

Run:
    python rag_warning_letters.py                  # build index + start chat
    python rag_warning_letters.py --skip-index     # skip rebuilding if index exists
"""

import os
import csv
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import pipeline

# ── Configuration ─────────────────────────────────────────────────────────────

# Load environmental variables
load_dotenv()

# Directories and paths
BASE_DIR         = Path(__file__).parent
LETTERS_DIR      = BASE_DIR / "fda_warning_letters" / "full_text"
FAISS_INDEX_PATH = BASE_DIR / "fda_faiss_index"
METADATA_CSV     = BASE_DIR / "fda_warning_letters" / "metadata.csv"

# HuggingFace token — reads from environment variable by default
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Embedding model — runs locally, no API key needed
# Small and fast; swap for "BAAI/bge-large-en-v1.5" for better quality
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM — free HuggingFace Inference API
# Good free options:
#   "tiiuae/falcon-7b-instruct"            (lightweight)
#   "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LLM_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Chunk settings — how the letters are split for indexing
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between chunks to preserve context

# How many chunks to retrieve per query
TOP_K = 5

SIMILARITY_THRESHOLD = 0.7  # lower score = more similar (L2 distance)

# ------------ Prompt template ------------

PROMPT = PromptTemplate.from_template(
    "You are an expert on FDA regulatory compliance and warning letters. "
    "Answer the question using ONLY the excerpts provided. "
    "For every factual claim, append a citation tag like [SOURCE: <filename>]. "
    "If the answer cannot be found in the excerpts, say: "
    "'I don't have enough information in the provided warning letters to answer that.'\n\n"
    "Excerpts:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (cite sources inline):"
)


# ------------ Result type ------------

@dataclass
class CitedAnswer:
    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)

    def pretty_print(self) -> None:
        """Print the answer and deduplicated source list to stdout."""
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(self.answer.strip())

        if self.sources:
            print("\n" + "-" * 70)
            print("SOURCES")
            print("-" * 70)
            seen: set[str] = set()
            for meta in self.sources:
                src = meta.get("source", "unknown")
                if src in seen:
                    continue
                seen.add(src)
                url = meta.get("url", "")
                print(f"  • {src}")
                if url:
                    print(f"    {url}")

        print("=" * 70 + "\n")


# ------------ Load Metadata ------------

def _load_csv_metadata() -> dict[str, dict[str, str]]:
    """
    Read metadata.csv and return a dict keyed by zero-padded letter ID.

    Date fields (date, posted_date) are normalised to ISO format (YYYY-MM-DD)
    across several input formats. Returns an empty dict if the CSV does not
    exist so the caller can degrade gracefully.
    """
    path = Path(METADATA_CSV)
    if not path.exists():
        print(f"  No metadata CSV found at {METADATA_CSV} — skipping.")
        return {}
    
    DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y%m%d"]

    def normalise_date(raw: str) -> str:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return raw.strip()
    
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_columns = [c for c in (reader.fieldnames or []) if c != "id"]
        print(f"  CSV columns: {csv_columns}")

        for row in reader:
            key = row.get("id", "").strip()
            if not key:
                continue
            record = {col: row.get(col, "").strip() for col in csv_columns}
            for date_col in ("date", "posted_date"):
                if record.get(date_col):
                    record[date_col] = normalise_date(record[date_col])
            rows[key] = record

    print(f"  Loaded {len(rows)} metadata rows from {METADATA_CSV}")
    return rows


def attach_metadata(docs: list[Document]) -> list[Document]:
    """
    Join metadata.csv fields onto each LangChain Document in-place.

    The letter ID is parsed from the filename stem (e.g. '00042_Acme...' → '00042')
    and used to look up the matching CSV row. Matched fields are added to
    doc.metadata without overwriting any fields already present. The source
    field is normalised to the bare filename for consistent citation display.
    """
    matched = unmatched = 0
    sidecar = _load_csv_metadata()

    def parse_filename_id(stem: str) -> str:
        """'00042_Exactech__Inc_...' → '00042'"""
        return stem.split("_")[0]

    for doc in docs:
        filename = Path(doc.metadata.get("source", "")).name
        key      = parse_filename_id(Path(filename).stem)
        record   = sidecar.get(key) or {}

        if record:
            matched += 1
        else:
            unmatched += 1

        for k, v in record.items():
            doc.metadata.setdefault(k, v)

        doc.metadata["source"] = filename

    if sidecar:
        print(f"  Metadata joined: {matched} matched, {unmatched} unmatched")
    return docs


# ------------ Step 1: Load documents ------------

def load_documents(letters_dir: str) -> list[Document]:
    """
    Load all .txt files from letters_dir as LangChain Documents.

    Uses LangChain's DirectoryLoader with a glob pattern so subdirectories
    are also covered. Each document's metadata.source is set to the file path
    by the loader before attach_metadata normalises it to the bare filename.
    """
    print()
    print('='*80)
    print(f"Step 1 - Loading warning letters from: {letters_dir}")
    print('='*80)
    loader = DirectoryLoader(
        letters_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} documents")
    # print(docs[0].page_content[:500])     # validated
    return docs


# ------------ Step 2: Split into chunks ------------

def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of Documents into fixed-size chunks for embedding.

    Uses RecursiveCharacterTextSplitter with paragraph-first separators so
    chunk boundaries fall at natural prose breaks where possible. The overlap
    prevents violation descriptions from being split across chunk boundaries.
    """
    print()
    print('='*80)
    print(f"Step 2 - Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    print('='*80)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\tSplit {len(docs)} warning letters into {len(chunks)} chunks.")
    return chunks


# ------------ Step 3: Build or load FAISS index ------------

def _embeddings() -> HuggingFaceEmbeddings:
    """
    Construct and return the sentence-transformers embedding model.

    Runs on CPU with normalised embeddings, which is required for cosine
    similarity to be computed correctly by FAISS.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(chunks: list[Document]) -> FAISS:
    """
    Embed chunks and build a FAISS index, then save it to disk.

    The index is saved to FAISS_INDEX_PATH so it can be reloaded on
    subsequent runs with --skip-index, avoiding re-embedding the corpus.
    """
    print()
    print('='*80)
    print(f"Step 3 - Building FAISS index using embeddings: {EMBEDDING_MODEL}")
    print('='*80)
    print("\t(This may take a few minutes for large collections...)")
    index = FAISS.from_documents(chunks, _embeddings(), distance_strategy="COSINE")
    index.save_local(str(FAISS_INDEX_PATH))
    print(f"\tIndex saved to: {FAISS_INDEX_PATH}/")
    return index


def load_index() -> FAISS:
    """
    Load a previously saved FAISS index from FAISS_INDEX_PATH.

    allow_dangerous_deserialization is required by FAISS when loading from
    a pickle-based format. Only load indexes from sources you trust.
    """
    print()
    print('='*80)
    print(f"Step 3 - Loading existing FAISS index from: {FAISS_INDEX_PATH}/")
    print('='*80)

    index = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )
    print("\tIndex loaded.")
    return index


# ------------ Step 4: Build Retrieval and Context Formatting ------------

def retrieve(index: FAISS, query: str) -> list[Document]:
    """
    Query the FAISS index and return the top-k most relevant chunks.

    Chunks with a cosine distance above SIMILARITY_THRESHOLD are filtered out
    before being returned. This avoids passing irrelevant context to the LLM
    when no letter closely matches the query.
    """
    print()
    print(f'Retrieving top {TOP_K} relevant documents.\n')

    results = index.similarity_search_with_score(query, k=TOP_K)

    # ------ COMMENTED OUT DUE TO LACK OF GOOD EMBEDDINGS AND SCORES -------
    # # Filter out chunks below the similarity threshold
    # filtered = [doc for doc, score in results if score < SIMILARITY_THRESHOLD]

    # if not filtered:
    #     print("  No chunks passed the similarity threshold.")
    # else:
    #     print(f"  Retrieved {len(filtered)}/{len(results)} chunks above threshold")

    # return filtered
    return [doc for doc, score in results]


def format_context(chunks: list[Document]) -> str:
    """Prefix each excerpt with a [SOURCE] header so the LLM can cite naturally."""
    parts = []
    for i, doc in enumerate(chunks, 1):
        m = doc.metadata
        header = "  |  ".join(filter(None, [
            f"[SOURCE: {m.get('source', f'chunk_{i}')}]",
            m.get("company"),
            m.get("date") or m.get("posted_date"),
            m.get("issuing_office"),
            m.get("subject"),
        ]))
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ------------ Step 5: LLM & QA ------------

def build_llm() -> HuggingFacePipeline:
    """
    Load the instruction-tuned LLM and wrap it in a LangChain pipeline.

    Uses SmolLM2-1.7B-Instruct, a small model that runs on CPU. Temperature
    is set low (0.1) to reduce hallucination on factual regulatory queries.
    Swap LLM_MODEL to use a larger model without changing anything else.
    """
    print()
    print('='*80)
    print(f'Step 4. Establishing connection to LLM:\t{LLM_MODEL}')
    print('='*80)

    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,     # return only generated tokens
        # repetition_penalty=1.3,  # penalises repeating the same token sequences
    )
    return HuggingFacePipeline(pipeline=pipe)


def ask(index: FAISS, llm: HuggingFacePipeline, question: str) -> CitedAnswer:
    """
    Run a single question through the full RAG pipeline and return a CitedAnswer.

    Retrieves relevant chunks, formats them with source headers, invokes the
    LLM, then strips the prompt echo from the response. The model sometimes
    repeats the excerpts or the question — common stop strings are used to
    truncate the answer before that happens.
    """
    chunks   = retrieve(index, question)

    if not chunks:
        return CitedAnswer(
            answer="I don't have enough information in the provided warning letters to answer that.",
            sources=[],
        )

    context  = format_context(chunks)
    raw = (PROMPT | llm).invoke({"context": context, "question": question})
    # print(f"DEBUG raw output:\n{raw[:500]}")  # remove after debugging

    # # HuggingFacePipeline returns prompt + completion — strip the prompt
    # marker = "Answer (cite sources inline):"
    # answer = raw.split(marker)[-1].strip()

    answer = raw.strip()
    # Truncate if the model starts repeating itself or dumps the excerpts back

    for stop in ["\n\nExcerpts:", "\n\nQuestion:", "\n\nReasoning:", "---", "\n\n\n"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return CitedAnswer(answer=answer, sources=[doc.metadata for doc in chunks])

# ------------ Step 6: Terminal chat loop ------------

def interactive_loop(index: FAISS, llm: HuggingFacePipeline) -> None:
    """
    Run a blocking terminal Q&A loop until the user exits.

    Accepts input at a 'Question>' prompt and prints a CitedAnswer after each
    query. Exits cleanly on 'q', 'quit', 'exit', EOF (Ctrl-D), or
    KeyboardInterrupt (Ctrl-C).
    """
    print("\n" + "=" * 70)
    print("FDA Warning Letters — RAG QA  (type 'quit', 'exit', or 'q' to exit)")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        print("\nThinking…")
        ask(index, llm, question).pretty_print()


def main() -> None:
    """
    Build or load the FAISS index and start the interactive Q&A loop.

    Intended for running rag_warning_letters.py directly. The canonical
    entry point for the full pipeline is 'python main.py chat'.
    """
    import argparse

    parser = argparse.ArgumentParser(description="FDA Warning Letters RAG Q&A")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Load existing FAISS index instead of rebuilding.",
    )
    args = parser.parse_args()

    if args.skip_index and FAISS_INDEX_PATH.exists():
        index = load_index()
    else:
        if not LETTERS_DIR.exists() or not any(LETTERS_DIR.iterdir()):
            print(
                f"Error: no letters found in '{LETTERS_DIR}'.\n"
                "Run 'python main.py scrape' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        docs   = load_documents(str(LETTERS_DIR))
        docs   = attach_metadata(docs)
        chunks = split_documents(docs)
        index  = build_index(chunks)

    interactive_loop(index, build_llm())


if __name__ == "__main__":
    import sys
    main()
