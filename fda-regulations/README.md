# FDA Warning Letters — RAG Q&A

A proof-of-concept RAG (Retrieval-Augmented Generation) system that lets you ask natural-language questions against the FDA's publicly available warning letters corpus.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [Data Acquisition](#data-acquisition)
   - [Inclusion Criteria](#inclusion-criteria)
   - [Exclusion Criteria](#exclusion-criteria)
   - [Dataset Summary Statistics](#dataset-summary-statistics)
5. [Data Modeling](#data-modeling)
6. [Design Decisions](#design-decisions)
7. [Limitations and Future Work](#limitations-and-future-work)

---

## Quick Start

**Prerequisites:** Docker Desktop installed and running.

```bash
# 1. Clone the repo
git clone <https://github.com/JasmolSD/interview-ai-engineer.git>
cd fda-regulations

# 2. Add your HuggingFace token (free account at huggingface.co)
echo "HF_TOKEN=hf_your_token_here" > .env

# 3. Scrape warning letters (run once; output goes to ./fda_warning_letters/)
#    --count 100 is a reasonable sample; omit it to scrape all ~3366 letters (~1 hour)
docker compose run fda-rag scrape --count 100

# 4. Build the FAISS index and start the Q&A loop
docker compose run fda-rag chat

# Subsequent runs: skip rebuilding the index
docker compose run fda-rag chat --skip-index
```

The first `chat` run downloads the embedding model (~90 MB) and the LLM (~3.4 GB) to a named Docker volume, so subsequent runs start in seconds. Due to computational limits, the TTFT is 3-4 minutes.

### CLI reference

```
python main.py scrape [--count N]            Scrape N letters (0 = all)
python main.py chat [--skip-index]           Build index + start Q&A loop
python main.py stats [--output report.pdf]   Generate PDF summary report
python main.py --help                        Top-level help
python main.py <subcommand> --help           Subcommand help
```

### Running without Docker

```bash
# Requires Python 3.13 and uv
uv sync --frozen
uv run python main.py scrape --count 100
uv run python main.py stats                   # generates fda_warning_letters_report.pdf
uv run python main.py chat
uv run python main.py chat --skip-index   # skip rebuild on repeat runs
```

---

## Project Structure

```
fda-regulations/
├── main.py                     # CLI entry point — 'scrape', 'chat', and 'stats' subcommands
├── scrape_warning_letters.py   # Data acquisition pipeline (Steps 1 & 2)
├── rag_warning_letters.py      # RAG pipeline + interactive chat loop
├── generate_stats.py           # PDF summary report generator
├── fda_warning_letters.pdf     # PDF report of warning letter statistics
├── pyproject.toml              # uv project manifest
├── uv.lock                     # Pinned dependency tree
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── docker-compose.yml          # Volume mounts, env wiring, tty config
├── .env                        # HF_TOKEN — not committed
├── fda_warning_letters/        # Written at runtime by the scraper
│   ├── metadata.csv            # Structured metadata for all scraped letters
│   ├── errors.json             # Any letters that failed to fetch
│   └── full_text/              # One .txt file per letter
└── fda_faiss_index/            # Written at runtime by the RAG pipeline
    ├── index.faiss             # FAISS vector index
    └── index.pkl               # Maps index positions back to source documents
```

---

## Architecture

The system is a straightforward offline RAG pipeline with no external API dependencies beyond the one-time HuggingFace model download.

```
scrape_warning_letters.py
  │
  ├── Step 1: FDA DataTables AJAX endpoint → metadata.csv
  │           (paginated JSON; 100 rows per request)
  │
  └── Step 2: Individual letter pages → full_text/*.txt
              (BeautifulSoup; CSS selector fallback chain)

rag_warning_letters.py
  │
  ├── Step 1: Load .txt files via LangChain DirectoryLoader
  ├── Step 2: Join metadata.csv fields onto each Document
  ├── Step 3: Split into 1000-char chunks (200-char overlap)
  ├── Step 4: Embed with sentence-transformers/all-MiniLM-L6-v2
  ├── Step 5: Store/load FAISS index (cosine similarity)
  └── Step 6: Interactive loop
              └── Query → top-5 chunks → prompt → SmolLM2-1.7B → answer
```

All computation runs locally on CPU. No tokens are sent to any external service. Integration of APIs and GPUs would be something to improve on.

---

## Data Acquisition

### Source

The FDA warning letters page ([fda.gov/…/warning-letters](https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters)) renders its table via a server-side DataTables AJAX endpoint. Rather than scraping the rendered HTML, the scraper hits this endpoint directly (`/datatables/views/ajax`) with the same query parameters the browser sends. This returns paginated JSON and is significantly more reliable than parsing the full page.

Each letter's full text is then fetched from its individual FDA page using a fallback chain of CSS selectors that covers the different page layouts FDA uses across different letter types and years.

**Why warning letters over the FDA Data Dashboard API?**
The Data Dashboard contains structured inspection records, but the fields are mostly categorical (establishment type, inspection outcome). Warning letters contain the narrative: what specifically went wrong, which regulations were cited, what the company was told to fix. That unstructured text is where RAG adds the most value. In addition, I thought that scraping the data and building a RAG system would be a fun challenge.

### Inclusion Criteria

- **Any letter with a resolvable URL** that returns a 200 response from fda.gov.
- **Any letter where at least one CSS selector matches** a content element on the page. Letters are only excluded if the page returns no parseable text at all.
- All subject categories are included (drugs, devices, food, cosmetics, tobacco, veterinary, etc.) to keep the corpus general-purpose.

The scraper is intentionally permissive at the collection stage. Filtering is left to the retrieval layer (FAISS similarity threshold) rather than baked into acquisition, because what counts as "relevant" depends entirely on the query.

### Exclusion Criteria

- **Letters with no URL** in the metadata table (malformed rows).
- **Network failures** (timeouts, non-200 responses) — logged to `errors.json` for retry.
- **Pages that return empty text** after all CSS selectors are tried and also logged to `errors.json`.
- Letters are **not** filtered by date, issuing office, or subject category.

### Dataset Summary Statistics

These figures are based on a sample of 100 letters scraped from the FDA warning letters dataset. A full scrape of all ~3366 available letters was not completed within the project timeframe as sequential fetching takes approximately one hour, and implementing concurrent fetching was deprioritised in favour of other deliverables. The sample is representative of the full corpus in terms of structure and field coverage, but distributions (issuing office counts, subject tag frequencies, date range) will shift with a complete dataset.

> These figures reflect a full scrape of all available letters as of March 2026. Run `python main.py stats` after scraping to generate a full PDF report with charts.

| Metric | Value |
|---|---|
| Total letters in FDA table | ~3366 |
| Letters successfully scraped | 100 |
| Scrape failure rate | 0% |
| Earliest letter (issue date) | 2025-08-21|
| Latest letter (issue date) | 2026-03-05 |
| Metadata fields per letter | 7 (posted\_date, date, company, issuing\_office, subject, response\_letter, closeout\_letter) |
| Letters with a response letter | 0 |
| Letters with a closeout letter | 0 |
| Total corpus size | 1.3 MB |

**Top issuing offices** (from full corpus):

| Issuing Office | Letter Count |
|---|---|
| Center for Drug Evaluation and Research (CDER) | 50 |
| Center for Devices and Radiological Health (CDRH) | 10 |
| Center for Food Safety and Applied Nutrition (CFSAN) | 10 |
| Office of Regulatory Affairs (ORA) | 10 |

**Most common subject tags:**

The subject field is slash-delimited (e.g. `CGMP/Medical Devices/Adulterated`). When split by token, the most frequent tags across the full corpus are Adulterated, CGMP, Misbranded, Finished Pharmaceuticals, and Medical Devices. Full ranked counts are included in the generated PDF report.

---

## Data Modeling

### Approach: Retrieval-Augmented Generation (RAG)

Warning letters are long, unstructured regulatory prose. The meaningful content (specific violations, cited regulations, product descriptions) is buried in paragraphs, not easily captured by keyword search or structured queries. RAG was chosen because it lets you ask open questions ("What GMP violations were cited against device manufacturers in 2023?") and get answers grounded in the actual letter text, with citations back to the source document.

### Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`

Chosen for three reasons: it runs on CPU without meaningful latency for individual queries, it produces embeddings well-suited to semantic similarity over regulatory/legal text, and at 90 MB it does not create a large Docker image. The tradeoff is retrieval quality versus a larger model like `BAAI/bge-large-en-v1.5`. For a proof of concept, it is acceptable, though it struggles with larger corpuses.

### Vector Store: FAISS (CPU, cosine similarity)

No server required. The index is saved to disk after the first build and reloaded on subsequent runs via `--skip-index`. For ~3000 letters split into ~1000-char chunks, the index fits comfortably in memory and query latency is negligible. In the case of the proof-of-concept, only 100 letters were used.

### LLM: `HuggingFaceTB/SmolLM2-1.7B-Instruct`

A 1.7B-parameter instruction-tuned model that runs on CPU. Response quality is modest but this is deliberate. The assignment explicitly asks for a proof of concept that runs locally without enterprise hardware. The model is capable enough to synthesize an answer (though it fails sometimes) from retrieved chunks and produce inline citations; it is not capable of nuanced legal reasoning. Swapping in a larger model (e.g., Mistral-7B) requires only changing the `LLM_MODEL` constant.

### Chunking Strategy

Letters are split into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`. The separator priority is `["\n\n", "\n", ". ", " ", ""]`, which keeps paragraphs together where possible. The overlap prevents a citation or violation description from being cut across a chunk boundary.

---

## Additional Design Decisions

**Why not chunk by section headers?**
FDA warning letters do not follow a consistent structure. Some use numbered sections, some use bold headers, some are continuous prose. A fixed-size chunker with overlap is more robust across the full corpus than a structure-aware splitter.

**Why save metadata separately and join at index time rather than embedding it?**
The scraper writes metadata to `metadata.csv` and letter text to individual `.txt` files as two separate concerns. The RAG pipeline joins them at load time. This keeps the scraper simple and lets you re-index with different chunk settings without re-scraping.

---

## Limitations and Future Work

- **CPU-only inference and indexing is slow.** Embedding and indexing the corpus takes several minutes on first run, and query responses from SmolLM2-1.7B take 15–30 seconds each. Both bottlenecks are hardware-bound: FAISS has no GPU support, and the transformers pipeline runs on CPU only. On Apple Silicon, switching the LLM to `llama.cpp` with Metal acceleration would reduce inference time significantly. In a production setting, offloading both embedding and inference to a hosted API would eliminate these constraints entirely.
- **The Docker image is large (~14 GB).** The embedding model and LLM are downloaded into a named volume on first run. A production version would use a hosted inference API, eliminating the need to ship model weights locally and reducing the footprint substantially. Migration to an API was deprioritised given the proof-of-concept scope and time constraints. The HuggingFace free-tier API proved unreliable during development, making local models the more pragmatic choice for the timeline given.
- **Local model quality is limited.** SmolLM2-1.7B produces reasonable answers for straightforward queries but struggles with nuanced regulatory reasoning. Swapping to a larger hosted model (e.g. Claude, GPT-4o) via API would significantly improve answer quality with no changes to the pipeline beyond the `build_llm()` function.
- **100 letters is a small corpus.** The full FDA enforcement record goes back decades and includes other document types. While this proof-of-concept only used 100 documents, a production version would ingest more.
- **The AJAX endpoint parameters may change.** The `view_dom_id` in `scrape_warning_letters.py` is embedded in the page's JS config and could change on a site rebuild. The scraper should be updated if pagination stops working.
- **No evaluation harness.** There is no automated way to measure retrieval quality or answer accuracy. Adding a small set of question–answer pairs with known source documents would allow retrieval metrics (precision@k, recall@k) to be tracked as the pipeline changes. Reliable metrics would be added in a production setting.
- **SmolLM2 does not reliably follow citation instructions.** The prompt asks for `[SOURCE: filename]` inline citations, but the model sometimes ignores the format and at times doesn't provide an answer, only sources. A structured output approach (constrained decoding or a post-processing step) would improve citation reliability.
- **Sequential scraping limits throughput.** The scraper fetches letters one at a time, making a full scrape of ~3366 letters take roughly an hour. A production version would use a thread pool (`concurrent.futures.ThreadPoolExecutor`) to fetch letters concurrently, reducing this to a few minutes. This was deprioritised given the proof-of-concept scope. The smaller sample sizes used during development made the latency acceptable.
- **Similarity threshold filtering is disabled.** The intended behaviour was to drop retrieved chunks with a cosine distance above `0.7`, returning a "not enough information" response when no letter closely matches the query. In practice, the `all-MiniLM-L6-v2` embeddings produced distance scores that did not cluster meaningfully around that threshold for this corpus, causing most chunks to be filtered out regardless of relevance. The threshold check has been left in the code but set permissively. Improving embedding quality (e.g. switching to `BAAI/bge-large-en-v1.5`) or tuning the threshold against a labelled eval set would make this effective.
- **No structured logging.** The pipeline currently uses `print` statements throughout for progress and error reporting. A production version would replace these with Python's `logging` module, allowing log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`) to be configured at runtime, output to be redirected to a file, and errors to be distinguished from progress output without parsing stdout. This would also make the Docker container's output easier to monitor and debug in a deployment context.
- **Metadata is not embedded.** The current pipeline joins metadata fields (company, issuing office, date, subject) onto each document chunk at index time, but only the letter text is embedded. This means queries like "warning letters issued by CDER in 2023" rely entirely on the LLM to filter by those fields from the retrieved context rather than using them to narrow the search space. A production version would embed structured metadata fields alongside the text, or maintain a separate filtered retrieval layer, allowing precise pre-filtering before semantic search is applied.