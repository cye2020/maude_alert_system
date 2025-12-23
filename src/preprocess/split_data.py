from pathlib import Path
import argparse
import polars as pl

from src.preprocess.config import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", '-w',
        type=int,
        nargs="+",
        default=[3, 2, 1],
        help="분할 가중치 리스트 (예: --weights 3 2 1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_config()

    data_dir = Path(cfg.base["paths"]["local"]["silver"])
    temp_dir = Path(cfg.base["paths"]["local"]["temp"])
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / "maude_preprocess_step1.parquet"

    lf = pl.scan_parquet(input_path)

    # 1) row 수 계산 (여기서만 실행)
    n_rows = lf.select(pl.len()).collect().item()

    weights = args.weights
    total_weight = sum(weights)

    base = n_rows // total_weight
    remainder = n_rows % total_weight

    offset = 0

    for i, w in enumerate(weights):
        size = w * base

        # remainder는 앞쪽 파트부터 1개씩 분배
        extra = min(remainder, w)
        size += extra
        remainder -= extra

        lf_part = lf.slice(offset, size)

        output_path = temp_dir / f"part_{i}.parquet"
        lf_part.sink_parquet(output_path)

        offset += size


if __name__ == "__main__":
    main()
