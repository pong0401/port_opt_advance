from __future__ import annotations

from collections.abc import Iterable


SHARE_CLASS_REPRESENTATIVES = {
    "GOOGL": "GOOG",
}


def canonical_share_class(ticker: str) -> str:
    clean = str(ticker).upper()
    return SHARE_CLASS_REPRESENTATIVES.get(clean, clean)


def drop_duplicate_share_classes(tickers: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        canonical = canonical_share_class(str(ticker))
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)
    return deduped


def drop_duplicate_share_classes_available(tickers: Iterable[str], available: Iterable[str]) -> list[str]:
    available_set = {str(ticker).upper() for ticker in available}
    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        clean = str(ticker).upper()
        canonical = canonical_share_class(clean)
        selected = canonical if canonical in available_set else clean
        if selected not in available_set or canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(selected)
    return deduped


def duplicate_share_classes(tickers: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for ticker in tickers:
        clean = str(ticker).upper()
        groups.setdefault(canonical_share_class(clean), [])
        if clean not in groups[canonical_share_class(clean)]:
            groups[canonical_share_class(clean)].append(clean)
    return {canonical: names for canonical, names in groups.items() if len(names) > 1}


def assert_no_duplicate_share_classes(tickers: Iterable[str]) -> None:
    duplicates = duplicate_share_classes(tickers)
    if not duplicates:
        return
    pairs = ", ".join(f"{canonical}: {'/'.join(names)}" for canonical, names in duplicates.items())
    raise ValueError(f"Duplicate share classes detected: {pairs}")
