from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataset image per kelas menjadi train/val/test."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset"),
        help="Folder dataset sumber, format: input_dir/<class_name>/*.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_split"),
        help="Folder hasil split: output_dir/{train,val,test}/<class_name>/*.jpg",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus output-dir lama jika sudah ada.",
    )
    return parser.parse_args()


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def copy_files(files: list[Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = target_dir / src.name
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("Total rasio train+val+test harus = 1.0")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset tidak ditemukan: {input_dir}")

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"Tidak ada folder kelas di: {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output dir {output_dir} sudah ada. Tambahkan --overwrite untuk menimpa."
            )
        shutil.rmtree(output_dir)

    random.seed(args.seed)
    summary: dict[str, dict[str, int]] = {}

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = sorted(
            [*class_dir.glob("*.jpg"), *class_dir.glob("*.jpeg"), *class_dir.glob("*.png")]
        )
        if not files:
            continue

        random.shuffle(files)
        train_n, val_n, test_n = split_counts(len(files), args.train_ratio, args.val_ratio)

        train_files = files[:train_n]
        val_files = files[train_n : train_n + val_n]
        test_files = files[train_n + val_n :]

        copy_files(train_files, output_dir / "train" / class_name)
        copy_files(val_files, output_dir / "val" / class_name)
        copy_files(test_files, output_dir / "test" / class_name)

        summary[class_name] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    print("Split selesai.")
    print(f"Input : {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print("-" * 50)
    for class_name, stats in summary.items():
        print(
            f"{class_name:>10} | total={stats['total']:>4} | "
            f"train={stats['train']:>4} | val={stats['val']:>4} | test={stats['test']:>4}"
        )


if __name__ == "__main__":
    main()
