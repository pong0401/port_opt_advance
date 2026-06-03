from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_streamlit_precomputed import _refresh_latest_effective_weight_files  # noqa: E402


def main() -> None:
    _refresh_latest_effective_weight_files()
    print("Updated PIT Step 2.5 latest effective weights from latest local SPY trend exposure.")


if __name__ == "__main__":
    main()
