"""
fda-regulations — CLI entry point
==================================
Two subcommands that map to the two phases of the pipeline:

    python main.py scrape          # Phase 1: collect letters from fda.gov
    python main.py chat            # Phase 2: build index (if needed) + Q&A loop

Run either subcommand with --help for all available options.
"""

import argparse
import sys


def cmd_scrape(args: argparse.Namespace) -> None:
    """
    Handler for the 'scrape' subcommand.

    Converts --count 0 (the CLI default meaning "no limit") to None before
    passing it to run_scraper, which treats None as "fetch everything".
    """
    from scrape_warning_letters import run_scraper

    count: int | None = args.count if args.count > 0 else None
    run_scraper(max_letters=count)


def cmd_chat(args: argparse.Namespace) -> None:
    """
    Handler for the 'chat' subcommand.

    Builds or loads the FAISS index, then starts the interactive Q&A loop.

    Index behaviour:
    - If --skip-index is passed and the index already exists on disk, it is
        loaded directly, skipping the embed step (saves several minutes).
    - Otherwise the full pipeline runs: load .txt files → attach metadata →
        chunk → embed → build FAISS index → save to disk.

    Exits early with a clear error if no scraped letters are found, since that
    is the most likely mistake on a first run.
    """
    from pathlib import Path
    from rag_warning_letters import (
        load_documents,
        attach_metadata,
        split_documents,
        build_index,
        load_index,
        build_llm,
        interactive_loop,
        FAISS_INDEX_PATH,
        LETTERS_DIR,
    )

    if args.skip_index and Path(FAISS_INDEX_PATH).exists():
        index = load_index()
    else:
        if not Path(LETTERS_DIR).exists() or not any(Path(LETTERS_DIR).iterdir()):
            print(
                f"Error: no letters found in '{LETTERS_DIR}'.\n"
                "Run 'python main.py scrape' first.",
                file=sys.stderr,
            )
            sys.exit(1)

        docs = load_documents(str(LETTERS_DIR))
        docs = attach_metadata(docs)
        chunks = split_documents(docs)
        index = build_index(chunks)

    interactive_loop(index, build_llm())


def cmd_stats(args: argparse.Namespace) -> None:
    """
    Handler for the 'stats' subcommand.

    Reads metadata.csv and the full_text directory and writes a PDF summary
    report containing corpus overview statistics and three charts:
    letters per year, top issuing offices, and top subject tags.

    Exits early if metadata.csv does not exist, since there is nothing to
    report before at least one scrape has been run.
    """
    from pathlib import Path
    from generate_stats import build_report
    import pandas as pd

    csv_path = Path(args.csv)
    text_dir = Path(args.text_dir)
    out_path = Path(args.output)

    if not csv_path.exists():
        print(
            f"Error: metadata CSV not found at '{csv_path}'.\n"
            "Run 'python main.py scrape' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {csv_path}")
    build_report(df, text_dir, out_path)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct and return the top-level argument parser.

    Subcommands:
        scrape  Paginate the FDA DataTables endpoint and fetch full letter text.
        chat    Embed the corpus into a FAISS index and open the Q&A loop.
        stats   Generate a PDF summary report from scraped metadata.    # ← add this

    Kept as a standalone function so it can be imported and tested independently
    of the main() entry point.
    """
    parser = argparse.ArgumentParser(
        prog="main",
        description="FDA Warning Letters — scrape and RAG Q&A pipeline",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── scrape ────────────────────────────────────────────────────────────────
    scrape_p = sub.add_parser(
        "scrape",
        help="Collect warning letters from fda.gov",
        description=(
            "Hits the FDA DataTables AJAX endpoint to collect metadata, then "
            "fetches the full text of each letter. Output is written to "
            "fda_warning_letters/metadata.csv and fda_warning_letters/full_text/."
        ),
    )
    scrape_p.add_argument(
        "--count",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Maximum number of letters to scrape. "
            "Defaults to 0, which means scrape all (~3 366). "
            "Use a small number (e.g. 50) for a quick test run."
        ),
    )
    scrape_p.set_defaults(func=cmd_scrape)

    # ── chat ──────────────────────────────────────────────────────────────────
    chat_p = sub.add_parser(
        "chat",
        help="Build the FAISS index (if needed) and start the Q&A loop",
        description=(
            "Loads scraped letters, embeds them with sentence-transformers, "
            "stores them in a FAISS index, then starts an interactive terminal "
            "session where you can ask questions about the corpus."
        ),
    )
    chat_p.add_argument(
        "--skip-index",
        action="store_true",
        help=(
            "Load an existing FAISS index from disk instead of rebuilding. "
            "Saves several minutes on repeat runs. Ignored if the index does not exist."
        ),
    )
    chat_p.set_defaults(func=cmd_chat)

    # ── stats ─────────────────────────────────────────────────────────────────
    stats_p = sub.add_parser(
        "stats",
        help="Generate a PDF summary report from the scraped metadata",
        description=(
            "Reads metadata.csv and the full_text directory and writes a PDF "
            "report with corpus overview statistics and charts. Run after scraping."
        ),
    )
    stats_p.add_argument(
        "--csv",
        default="fda_warning_letters/metadata.csv",
        help="Path to metadata CSV (default: fda_warning_letters/metadata.csv)",
    )
    stats_p.add_argument(
        "--text-dir",
        default="fda_warning_letters/full_text",
        help="Path to full_text directory (default: fda_warning_letters/full_text)",
    )
    stats_p.add_argument(
        "--output",
        default="fda_warning_letters_report.pdf",
        help="Output PDF path (default: fda_warning_letters_report.pdf)",
    )
    stats_p.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
