"""Quick sanity check: build evidence for the first 5 rows of dev.csv and inspect."""

from __future__ import annotations

import pandas as pd

from preprocess import build_evidence_column


def main() -> None:
    df = pd.read_csv("dev.csv").head(5)
    df_ev = build_evidence_column(
        df, pdf_dir="paper_evidence/dev/", top_k=3, max_chars=1000,
    )

    for row in df_ev.itertuples(index=False):
        evidence = row.evidence or ""
        print("=" * 80)
        print(f"id:            {row.id}")
        print(f"paper_id:      {row.paper_id}")
        print(f"text[:100]:    {str(row.text)[:100]}")
        print(f"evidence:      {evidence}")
        print(f"evidence len:  {len(evidence)} chars")

    empty_count = sum(1 for ev in df_ev["evidence"] if not ev)
    print("=" * 80)
    print(f"Empty evidence rows: {empty_count} / {len(df_ev)}")


if __name__ == "__main__":
    main()
