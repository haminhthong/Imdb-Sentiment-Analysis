"""Tạo fixture nhỏ cho test/smoke; không được dùng thay IMDB chính thức."""

import argparse
from pathlib import Path

import pandas as pd

SMOKE_ROWS = [
    {"text": "A genuinely wonderful movie with great acting.", "label": 1},
    {"text": "An awful and boring film with a weak story.", "label": 0},
    {"text": "Excellent direction and memorable performances.", "label": 1},
    {"text": "Terrible pacing made this movie unpleasant.", "label": 0},
    {"text": "A beautiful, engaging and thoughtful review.", "label": 1},
    {"text": "A dull, predictable and disappointing experience.", "label": 0},
]


def main() -> None:
    """Ghi dữ liệu smoke vào thư mục fixture riêng."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/fixtures/imdb_smoke/train.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SMOKE_ROWS).to_csv(output, index=False, encoding="utf-8")
    print(f"Created smoke dataset at: {output.resolve()}")


if __name__ == "__main__":
    main()
