#!/usr/bin/env python3
"""Export sklearn breast_cancer dataset to CSV with a `target` column.

target convention (matches sklearn): 0 = malignant, 1 = benign.
All 30 feature columns keep their original human-readable names (with spaces),
so the Rust program can address them by name, e.g. "worst radius".
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

OUT = Path("/home/z/my-project/breast_cancer_rust/data/breast_cancer.csv")


def main() -> None:
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()  # features + target
    # sanity: 0 malignant, 1 benign, 569 rows, 30 features + target
    assert df.shape == (569, 31), df.shape
    assert set(df["target"].unique()) == {0, 1}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}  cols={len(df.columns)}")

    # Verify the two reference ratios the task expects.
    for f in ("worst radius", "worst texture"):
        means = df.groupby("target")[f].mean()
        ratio = means[0] / means[1]
        print(f"{f}: malignant={means[0]:.4f} benign={means[1]:.4f} "
              f"ratio={ratio:.4f} rounded={round(ratio, 2)}")


if __name__ == "__main__":
    main()
