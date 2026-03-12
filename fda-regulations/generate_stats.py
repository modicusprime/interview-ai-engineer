"""
FDA Warning Letters — Dataset Summary Report (PDF)
===================================================
Reads metadata.csv and the full_text directory, then writes a self-contained
PDF report with summary statistics and charts.

Usage:
    uv run python generate_stats.py
    uv run python generate_stats.py --csv path/to/metadata.csv \\
                                    --text-dir path/to/full_text \\
                                    --output report.pdf

Dependencies (all already in pyproject.toml or stdlib):
    reportlab, matplotlib, pandas
"""

import argparse
import io
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

matplotlib.use("Agg")  # headless — no display needed

# ------------ Configuration ------------

BASE_DIR         = Path(__file__).parent
DEFAULT_TEXT_DIR = BASE_DIR / "fda_warning_letters" / "full_text"
DEFAULT_CSV      = BASE_DIR / "fda_warning_letters" / "metadata.csv"
DEFAULT_OUTPUT   = BASE_DIR / "fda_warning_letters_report.pdf"
DATE_FORMATS     = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"]
TOP_N            = 10

# Palette — two greys and one FDA-adjacent blue
BLUE  = "#1a4f8a"
LGREY = "#f4f4f4"
DGREY = "#333333"


# ------------ Style helpers ------------

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=24,
            textColor=colors.HexColor(BLUE),
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor(DGREY),
            spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=14,
            textColor=colors.HexColor(BLUE),
            spaceBefore=18,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            textColor=colors.HexColor(DGREY),
            leading=15,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.grey,
            alignment=1,  # centre
        ),
    }


def _divider() -> HRFlowable:
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor(BLUE), spaceAfter=6)


def _kv_table(rows: list[tuple[str, str]], styles: dict) -> Table:
    """Two-column key–value table used for the overview stats block."""
    data = [[Paragraph(f"<b>{k}</b>", styles["body"]), Paragraph(v, styles["body"])]
            for k, v in rows]
    t = Table(data, colWidths=[3.2 * inch, 3.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LGREY)),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
            [colors.HexColor(LGREY), colors.white]),
        ("BOX",     (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("GRID",    (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    return t


def _chart_image(fig: matplotlib.figure.Figure, width: float = 6.4) -> Image:
    """Render a matplotlib figure to an in-memory PNG and return a ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    # Preserve aspect ratio
    w_pt  = width * inch
    aspect = fig.get_figheight() / fig.get_figwidth()
    return Image(buf, width=w_pt, height=w_pt * aspect)


# ------------ Chart builders ------------

def _chart_letters_per_year(parsed_dates: pd.Series) -> matplotlib.figure.Figure:
    years  = parsed_dates.map(lambda d: d.year).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(years.index.astype(str), years.to_numpy(), color=BLUE, width=0.6)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Letters", fontsize=9)
    ax.set_title("Letters Issued per Year", fontsize=11, color=DGREY)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate each bar with its exact count
    for bar, count in zip(bars, years.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + years.max() * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=7,
            color=DGREY,
        )
    ax.set_ylim(top=years.max() * 1.12)  # make room for labels

    fig.tight_layout()
    return fig


def _chart_top_offices(df: pd.DataFrame) -> matplotlib.figure.Figure:
    counts = (
        df["issuing_office"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(TOP_N)
        .iloc[::-1]  # horizontal bars read top-to-bottom
    )
    # Shorten long labels for readability
    labels = [l if len(l) <= 45 else l[:42] + "…" for l in counts.index]

    fig, ax = plt.subplots(figsize=(7, max(3, len(counts) * 0.45)))
    bars = ax.barh(labels, counts.to_numpy(), color=BLUE)
    ax.set_xlabel("Letter count", fontsize=9)
    ax.set_title(f"Top {TOP_N} Issuing Offices", fontsize=11, color=DGREY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # Annotate each bar with its exact count
    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_width() + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=8,
            color=DGREY,
        )
    ax.set_xlim(right=counts.max() * 1.12)  # make room for labels

    fig.tight_layout()
    return fig


def _chart_top_subjects(df: pd.DataFrame) -> matplotlib.figure.Figure:
    tokens = (
        df["subject"]
        .dropna()
        .str.split("/")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    counts = tokens.value_counts().head(TOP_N).iloc[::-1]
    labels = [l if len(l) <= 45 else l[:42] + "…" for l in counts.index]

    fig, ax = plt.subplots(figsize=(7, max(3, len(counts) * 0.45)))
    bars = ax.barh(labels, counts.to_numpy(), color=BLUE)
    ax.set_xlabel("Occurrences", fontsize=9)
    ax.set_title(f"Top {TOP_N} Subject Tags", fontsize=11, color=DGREY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # Annotate each bar with its exact count
    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_width() + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=8,
            color=DGREY,
        )
    ax.set_xlim(right=counts.max() * 1.12)  # make room for labels

    fig.tight_layout()
    return fig


# ------------ Date helper ------------

def _parse_date(raw: str) -> datetime | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


# ------------ Report assembly ------------

def build_report(df: pd.DataFrame, text_dir: Path, output_path: Path) -> None:
    doc    = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )
    styles = _styles()
    story: list = []

    # Cover / header 
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("FDA Warning Letters", styles["title"]))
    story.append(Paragraph("Dataset Summary Report", styles["subtitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}  ·  "
        f"Source: {DEFAULT_CSV}",
        styles["caption"],
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_divider())

    # Corpus overview 
    txt_files   = list(text_dir.glob("*.txt")) if text_dir.exists() else []
    total_meta  = len(df)
    total_files = len(txt_files)
    missing     = total_meta - total_files
    fail_pct    = f"{missing / total_meta * 100:.1f}%" if total_meta else "—"

    parsed_dates = df["date"].dropna().map(_parse_date).dropna()
    date_min = parsed_dates.min().strftime("%Y-%m-%d") if not parsed_dates.empty else "—"
    date_max = parsed_dates.max().strftime("%Y-%m-%d") if not parsed_dates.empty else "—"
    span     = (parsed_dates.max() - parsed_dates.min()).days if not parsed_dates.empty else "—"

    has_response = df["response_letter"].replace("", pd.NA).notna().sum()
    has_closeout = df["closeout_letter"].replace("", pd.NA).notna().sum()

    story.append(Paragraph("Corpus Overview", styles["h1"]))
    story.append(_kv_table([
        ("Rows in metadata.csv",           str(total_meta)),
        ("Full-text files on disk",         str(total_files)),
        ("Failed to scrape",               f"{missing}  ({fail_pct})"),
        ("Earliest letter (issue date)",    date_min),
        ("Latest letter (issue date)",      date_max),
        ("Span (days)",                     str(span)),
        ("Letters with response letter",   f"{has_response}  ({has_response/total_meta*100:.1f}%)"),
        ("Letters with closeout letter",   f"{has_closeout}  ({has_closeout/total_meta*100:.1f}%)"),
    ], styles))

    # File size stats
    if txt_files:
        sizes = pd.Series([f.stat().st_size for f in txt_files])
        story.append(Spacer(1, 0.15 * inch))
        story.append(_kv_table([
            ("Mean file size",   f"{sizes.mean():,.0f} bytes"),
            ("Median file size", f"{sizes.median():,.0f} bytes"),
            ("Smallest file",    f"{sizes.min():,.0f} bytes"),
            ("Largest file",     f"{sizes.max():,.0f} bytes"),
            ("Total corpus size",f"{sizes.sum()/1_000_000:.1f} MB"),
        ], styles))

    # Letters per year chart
    if not parsed_dates.empty:
        story.append(PageBreak())
        story.append(Paragraph("Letters Issued Over Time", styles["h1"]))
        story.append(_divider())
        story.append(Paragraph(
            "Volume of warning letters by year based on the letter issue date "
            "(not the posted date, which lags by weeks to months).",
            styles["body"],
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_chart_image(_chart_letters_per_year(parsed_dates)))
        story.append(Paragraph(
            "Figure 1 — Warning letters issued per calendar year.",
            styles["caption"],
        ))

    # Issuing offices chart
    if df["issuing_office"].replace("", pd.NA).notna().any():
        story.append(Spacer(1, 0.25 * inch))
        story.append(Paragraph("Issuing Offices", styles["h1"]))
        story.append(_divider())
        story.append(Paragraph(
            f"The {TOP_N} FDA centres and offices responsible for the most warning letters "
            "in the scraped corpus.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_chart_image(_chart_top_offices(df)))
        story.append(Paragraph(
            f"Figure 2 — Top {TOP_N} issuing offices by letter count.",
            styles["caption"],
        ))

    # Subject tags chart
    if df["subject"].replace("", pd.NA).notna().any():
        story.append(PageBreak())
        story.append(Paragraph("Subject Categories", styles["h1"]))
        story.append(_divider())
        story.append(Paragraph(
            "Subject strings are slash-delimited (e.g. CGMP/Medical Devices/Adulterated). "
            "Each token is counted individually so common violation types are ranked "
            "across all letter categories.",
            styles["body"],
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_chart_image(_chart_top_subjects(df)))
        story.append(Paragraph(
            f"Figure 3 — Top {TOP_N} subject tags by occurrence.",
            styles["caption"],
        ))

    # Build
    doc.build(story)
    print(f"Report saved to: {output_path}")


# ------------ Entry point ------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a PDF summary report for the scraped FDA warning letters dataset."
    )
    parser.add_argument("--csv",      default=str(DEFAULT_CSV),      help=f"Path to metadata CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR), help=f"Path to full_text directory (default: {DEFAULT_TEXT_DIR})")
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT),   help=f"Output PDF path (default: {DEFAULT_OUTPUT})")
    return parser


def main() -> None:
    args     = build_parser().parse_args()
    csv_path = Path(args.csv)
    text_dir = Path(args.text_dir)
    out_path = Path(args.output)

    if not csv_path.exists():
        print(f"Error: metadata CSV not found at '{csv_path}'.")
        print("Run 'python main.py scrape' first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {csv_path}")
    build_report(df, text_dir, out_path)


if __name__ == "__main__":
    main()