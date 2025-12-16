import argparse
from src.loading import DataLoader

def main():
    parser = argparse.ArgumentParser(description="DataLoader CLI")

    parser.add_argument("--name", "-n", type=str, default='event',
                        help="Raw Data Name")
    parser.add_argument("--start", "-s", type=int,
                        help="Start year (e.g., 2020)")
    parser.add_argument("--end", "-e", type=int,
                        help="End year (e.g., 2025)")
    parser.add_argument("--output-file", "-o", type=str, required=True,
                        help="Output file path (e.g., output.parquet)")
    parser.add_argument("--max-workers", "-w", type=int, default=4,
                        help="Maximum number of workers")
    parser.add_argument("--skip", action="store_true",
                        help="Skip processing if schema exists")

    args = parser.parse_args()

    loader = DataLoader(
        name=args.name,
        start=args.start,
        end=args.end,
        output_file=args.output_file,
        max_workers=args.max_workers
    )

    loader.process(skip=args.skip)

if __name__=='__main__':
    main()