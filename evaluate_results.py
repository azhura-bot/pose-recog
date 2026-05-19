from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buat ringkasan evaluasi model dari metrics.json + classification_report.txt."
    )
    parser.add_argument("--metrics", type=Path, default=Path("models/lstm_pose/metrics.json"))
    parser.add_argument(
        "--classification-report",
        type=Path,
        default=Path("models/lstm_pose/classification_report.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/lstm_pose"))
    parser.add_argument("--overfit-gap-threshold", type=float, default=0.05)
    return parser.parse_args()


def resolve_existing_file(path: Path, label: str) -> Path:
    if path.exists() and path.is_file():
        return path

    candidates = sorted(Path(".").glob(f"**/{path.name}"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"{label} tidak ditemukan: {path}\n"
        "Pastikan proses training sudah dijalankan dan menghasilkan file evaluasi.\n"
        "Contoh:\n"
        "python train_lstm_pose.py --data-dir dataset_split --epochs 50 --batch-size 32 --learning-rate 0.0005 --patience 12 --output-dir models/lstm_pose"
    )


def parse_classification_report(report_text: str, class_names: list[str]) -> dict[str, dict[str, float]]:
    metrics_by_class: dict[str, dict[str, float]] = {}

    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            continue

        label = parts[0]
        if label not in class_names:
            continue

        try:
            precision, recall, f1_score, support = map(float, parts[1:])
        except ValueError:
            continue

        metrics_by_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "support": support,
        }

    missing = [name for name in class_names if name not in metrics_by_class]
    if missing:
        raise ValueError(f"Kelas berikut tidak ditemukan di classification_report: {missing}")

    return metrics_by_class


def diagnose_fitting(history: dict[str, list[float]], threshold: float) -> dict[str, float | str]:
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    final_train_acc = float(train_acc[-1])
    final_val_acc = float(val_acc[-1])
    final_train_loss = float(train_loss[-1])
    final_val_loss = float(val_loss[-1])

    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss

    if final_train_acc < 0.8 and final_val_acc < 0.8:
        diagnosis = "indikasi underfitting"
    elif acc_gap > threshold and loss_gap > 0:
        diagnosis = "indikasi overfitting"
    else:
        diagnosis = "fit cukup baik (tidak ada indikasi kuat overfitting/underfitting)"

    return {
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "acc_gap": float(acc_gap),
        "loss_gap": float(loss_gap),
        "diagnosis": diagnosis,
    }


def save_class_metrics_chart(
    metrics_by_class: dict[str, dict[str, float]], class_names: list[str], output_path: Path
) -> None:
    x = np.arange(len(class_names))
    width = 0.24

    precision = [metrics_by_class[name]["precision"] for name in class_names]
    recall = [metrics_by_class[name]["recall"] for name in class_names]
    f1 = [metrics_by_class[name]["f1"] for name in class_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p = ax.bar(x - width, precision, width, label="Precision", color="#1f77b4")
    bars_r = ax.bar(x, recall, width, label="Recall", color="#ff7f0e")
    bars_f = ax.bar(x + width, f1, width, label="F1-score", color="#2ca02c")

    ax.set_title("Per-Class Metrics (Testing Set)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for group in (bars_p, bars_r, bars_f):
        for rect in group:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def save_evaluation_summary(
    metrics: dict,
    fitting: dict[str, float | str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Model Evaluation Summary",
        "",
        "## Core Test Metrics",
        f"- Test accuracy: {metrics['test_accuracy']:.4f}",
        f"- Test macro F1: {metrics['test_macro_f1']:.4f}",
        f"- Test loss: {metrics['test_loss']:.4f}",
        "",
        "## Data Split",
        f"- Train: {metrics['num_samples']['train']} samples",
        f"- Validation: {metrics['num_samples']['val']} samples",
        f"- Test: {metrics['num_samples']['test']} samples",
        "",
        "## Fitting Diagnosis",
        f"- Final train accuracy: {fitting['final_train_acc']:.4f}",
        f"- Final validation accuracy: {fitting['final_val_acc']:.4f}",
        f"- Accuracy gap (train - val): {fitting['acc_gap']:.4f}",
        f"- Final train loss: {fitting['final_train_loss']:.4f}",
        f"- Final validation loss: {fitting['final_val_loss']:.4f}",
        f"- Loss gap (val - train): {fitting['loss_gap']:.4f}",
        f"- Kesimpulan: {fitting['diagnosis']}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_path = resolve_existing_file(args.metrics, "metrics.json")
    report_path = resolve_existing_file(
        args.classification_report, "classification_report.txt"
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")

    class_names: list[str] = metrics["class_names"]
    per_class = parse_classification_report(report_text, class_names)
    fitting = diagnose_fitting(metrics["history"], args.overfit_gap_threshold)

    save_class_metrics_chart(per_class, class_names, args.output_dir / "per_class_metrics_test.png")
    save_evaluation_summary(metrics, fitting, args.output_dir / "evaluation_summary.md")

    evaluation_json = {
        "test_metrics": {
            "accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["test_macro_f1"],
            "loss": metrics["test_loss"],
        },
        "fitting_diagnosis": fitting,
        "per_class_metrics": per_class,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_json, indent=2), encoding="utf-8"
    )

    print("Artifacts generated:")
    print(f"- {(args.output_dir / 'evaluation_summary.md').resolve()}")
    print(f"- {(args.output_dir / 'evaluation_summary.json').resolve()}")
    print(f"- {(args.output_dir / 'per_class_metrics_test.png').resolve()}")


if __name__ == "__main__":
    main()
