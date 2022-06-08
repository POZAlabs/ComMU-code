import argparse
from multiprocessing import cpu_count
from pathlib import Path

from conmu.preprocessor import PreprocessPipeline

def get_root_parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser("dataset preprocessing", add_help=True)
    root_parser.add_argument("--root_dir", type=str, required=True, help="root directory containing 'raw' directory")
    root_parser.add_argument("--csv_path", type=str, required=True, help="csv file path containing meta info")
    root_parser.add_argument("--num_cores", type=int, default=max(1, cpu_count() - 4))
    return root_parser


def main(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir).expanduser()
    pipeline = PreprocessPipeline()
    pipeline(
        root_dir=root_dir,
        csv_path=args.csv_path,
        num_cores=args.num_cores,
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = get_root_parser()
    known_args, _ = parser.parse_known_args()
    main(known_args)
