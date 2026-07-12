from __future__ import annotations

import json

import pandas as pd

from qqq_combo_gtaa import RESULT_PREFIX, ROOT, STRATEGY, latest_security_weights, load_prices


def write_outputs(latest: pd.DataFrame, sleeve: pd.DataFrame, meta: dict[str, object]) -> None:
    for output_dir in [ROOT / "result", ROOT / "data" / "precomputed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        latest.to_csv(output_dir / f"{RESULT_PREFIX}_latest_effective_weights.csv", index=False)
        sleeve.to_csv(output_dir / f"{RESULT_PREFIX}_latest_sleeve_weights.csv", index=False)
        (output_dir / f"{RESULT_PREFIX}_latest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        pd.DataFrame([meta]).to_csv(output_dir / f"{RESULT_PREFIX}_latest_meta.csv", index=False)


def main() -> None:
    prices = load_prices(refresh=True)
    latest, sleeve, meta = latest_security_weights(prices)
    write_outputs(latest, sleeve, meta)
    print(f"Updated {STRATEGY} latest weights through {meta['Latest Cache Trading Date']}")
    print(latest[["Asset", "Target Weight %", "Daily Exposure", "Effective Weight %"]].to_string(index=False))


if __name__ == "__main__":
    main()
