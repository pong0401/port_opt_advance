from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

LISTING_DIR = Path(__file__).resolve().parent.parent / "data" / "thai_stock" / "set100_listing"
OUTPUT_FILE = LISTING_DIR.parent / "set100_ticker_start_end.csv"
POWERSHELL = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
XLS_READER_SCRIPT = Path(__file__).resolve().with_name("read_set100_xls.ps1")


try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - helper script only
    raise SystemExit(
        "This script requires `pypdf`. Run it with the bundled Codex runtime Python or install pypdf locally."
    ) from exc


STOP_TOKENS = {
    "NO",
    "SYMBOL",
    "SET100",
    "SET50",
    "COMPANY",
    "SECTOR",
    "PCL",
    "PUBLIC",
    "LIMITED",
    "RANK",
    "BY",
    "ALPHABET",
    "INDEX",
    "CONSTITUENTS",
}


@dataclass(frozen=True)
class SnapshotWindow:
    start_date: str
    end_date: str
    label: str


def snapshot_window_from_name(path: Path) -> SnapshotWindow:
    name = path.name.upper()
    year_match = re.search(r"(20\d{2})", name)
    if not year_match:
        raise ValueError(f"Could not infer year from file name: {path.name}")
    year = int(year_match.group(1))

    if "H1" in name:
        return SnapshotWindow(f"{year}-01-01", f"{year}-06-30", f"H1 {year}")
    if "H2" in name or "2H" in name:
        return SnapshotWindow(f"{year}-07-01", f"{year}-12-31", f"H2 {year}")
    raise ValueError(f"Could not infer half-year window from file name: {path.name}")


def normalize_symbol(raw: object) -> str:
    text = str(raw or "").upper().strip()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = text.replace("*", "")
    text = re.sub(r"[^A-Z0-9.\-]", "", text)
    if not re.search(r"[A-Z0-9]", text):
        return ""
    if not text or text in STOP_TOKENS:
        return ""
    return text if text.endswith(".BK") else f"{text}.BK"


def is_compact_symbol(value: str) -> bool:
    base = value.removesuffix(".BK")
    return 1 <= len(base) <= 8


def extract_symbols_from_pdf(path: Path) -> list[str]:
    pages = [(page.extract_text() or "") for page in PdfReader(str(path)).pages]
    relevant_pages: list[str] = []
    anchor_index: int | None = None
    for idx, page_text in enumerate(pages):
        upper = page_text.upper()
        if "SET100" in upper and "CONSTITUENTS" in upper:
            anchor_index = idx
            break
        if "NO SYMBOL SET100" in upper:
            anchor_index = idx
            break

    if anchor_index is None:
        relevant_pages = pages
    else:
        relevant_pages = pages[anchor_index:]
        preview_text = "\n".join(relevant_pages[:2])
        preview_matches = re.findall(r"(?<![A-Z0-9])(\d{1,3})\s+([A-Z][A-Z0-9&.\-]{0,15})\b", preview_text)
        preview_ranks = {int(rank) for rank, _ in preview_matches if rank.isdigit()}
        if 1 not in preview_ranks and anchor_index > 0:
            relevant_pages = [pages[anchor_index - 1], *relevant_pages]

    text = "\n".join(relevant_pages)
    matches = re.findall(r"(?<![A-Z0-9])(\d{1,3})\s+([A-Z][A-Z0-9&.\-]{0,15})\b", text)
    ordered: list[str] = []
    seen: set[str] = set()
    for rank_text, raw_symbol in matches:
        rank = int(rank_text)
        if rank < 1 or rank > 100:
            continue
        symbol = normalize_symbol(raw_symbol)
        if not symbol or symbol in seen:
            continue
        ordered.append(symbol)
        seen.add(symbol)
        if len(ordered) == 100:
            break
    return ordered


def run_powershell_file(script_path: Path, *args: str) -> str:
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    )
    return result.stdout.strip()


def extract_symbols_from_xls(path: Path) -> list[str]:
    raw_json = run_powershell_file(XLS_READER_SCRIPT, str(path.resolve()))
    if not raw_json:
        return []
    data = json.loads(raw_json)
    if isinstance(data, dict):
        data = [data]

    symbols: list[str] = []
    seen: set[str] = set()
    for row in data:
        values = [str(value or "").strip() for value in row.values()]
        if not any(values):
            continue
        symbol = ""
        rank_idx = next((idx for idx, value in enumerate(values) if re.fullmatch(r"\d{1,3}", value)), None)
        if rank_idx is not None:
            rank = int(values[rank_idx])
            if rank < 1 or rank > 100:
                continue
            search_values = values[rank_idx + 1 :]
        else:
            search_values = values
        for candidate in search_values:
            normalized = normalize_symbol(candidate)
            if normalized and is_compact_symbol(normalized):
                symbol = normalized
                break
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) == 100:
            break
    return symbols


def extract_snapshot_symbols(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        return extract_symbols_from_pdf(path)
    if path.suffix.lower() == ".xls":
        return extract_symbols_from_xls(path)
    return []


def merge_membership_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (row["ticker"], row["start_date"], row["end_date"]))
    merged: list[dict[str, str]] = []
    for row in ordered:
        if not merged or merged[-1]["ticker"] != row["ticker"]:
            merged.append(dict(row))
            continue

        previous = merged[-1]
        prev_end_ts = pd.Timestamp(previous["end_date"])
        row_start_ts = pd.Timestamp(row["start_date"])
        if row_start_ts <= prev_end_ts + pd.Timedelta(days=1):
            previous["end_date"] = max(previous["end_date"], row["end_date"])
            previous["source_files"] = f"{previous['source_files']} | {row['source_files']}"
        else:
            merged.append(dict(row))
    return merged


def main() -> None:
    files = sorted(path for path in LISTING_DIR.iterdir() if path.suffix.lower() in {".xls", ".pdf"})
    if not files:
        raise SystemExit(f"No listing files found under {LISTING_DIR}")

    snapshot_rows: list[dict[str, str]] = []
    diagnostics: list[tuple[str, int, str]] = []
    for path in files:
        window = snapshot_window_from_name(path)
        symbols = extract_snapshot_symbols(path)
        diagnostics.append((path.name, len(symbols), window.label))
        if not symbols:
            continue
        for symbol in symbols:
            snapshot_rows.append(
                {
                    "ticker": symbol,
                    "start_date": window.start_date,
                    "end_date": window.end_date,
                    "source_files": path.name,
                }
            )

    if not snapshot_rows:
        raise SystemExit("No SET100 memberships were parsed from the source files.")

    merged = pd.DataFrame(merge_membership_rows(snapshot_rows)).sort_values(["ticker", "start_date"]).reset_index(drop=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Wrote {len(merged):,} merged membership intervals to {OUTPUT_FILE.name}")
    for file_name, count, label in diagnostics:
        print(f"{file_name}: {count} symbols ({label})")


if __name__ == "__main__":
    main()
