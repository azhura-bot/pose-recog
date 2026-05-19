from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment dataset pose dengan variasi brightness/contrast secara terkontrol."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset"),
        help="Folder dataset sumber: input_dir/<class_name>/*.jpg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_aug"),
        help="Folder output hasil augment.",
    )
    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=2,
        help="Jumlah varian augment baru per gambar asli.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed random untuk hasil yang reproducible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus output-dir dulu jika sudah ada.",
    )
    return parser.parse_args()


def adjust_brightness_contrast(img: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    # alpha mengatur kontras, beta mengatur brightness.
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def list_images(class_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        for item in sorted(path.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        path.rmdir()

    path.mkdir(parents=True, exist_ok=True)


def copy_original_image(src: Path, dst: Path) -> None:
    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {src}")
    cv2.imwrite(str(dst), image)


def augment_image(src: Path, dst: Path, rng: random.Random) -> None:
    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {src}")

    brightness_shift = rng.randint(-65, 20)
    contrast_scale = rng.uniform(0.85, 1.15)
    gamma = rng.uniform(0.75, 1.05)

    out = adjust_brightness_contrast(image, alpha=contrast_scale, beta=brightness_shift)
    out = adjust_gamma(out, gamma=gamma)

    cv2.imwrite(str(dst), out)


def main() -> None:
    args = parse_args()

    if args.variants_per_image < 1:
        raise ValueError("variants-per-image minimal 1")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dataset tidak ditemukan: {args.input_dir}")

    ensure_clean_output(args.output_dir, args.overwrite)

    rng = random.Random(args.seed)
    class_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"Tidak ada folder kelas di: {args.input_dir}")

    print(f"Input : {args.input_dir.resolve()}")
    print(f"Output: {args.output_dir.resolve()}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = list_images(class_dir)
        if not files:
            print(f"[SKIP] {class_name}: tidak ada gambar")
            continue

        out_class_dir = args.output_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_path in enumerate(files):
            stem = img_path.stem
            ext = ".jpg"

            original_name = f"{stem}_orig{ext}"
            copy_original_image(img_path, out_class_dir / original_name)

            for k in range(args.variants_per_image):
                aug_name = f"{stem}_aug{k + 1}{ext}"
                augment_image(img_path, out_class_dir / aug_name, rng)

        total_out = len(files) * (1 + args.variants_per_image)
        print(
            f"[OK] {class_name}: original={len(files)} | augmented={len(files) * args.variants_per_image} | total={total_out}"
        )

    print("Selesai augment dataset.")


if __name__ == "__main__":
    main()
