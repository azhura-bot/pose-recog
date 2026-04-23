from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model LSTM (PyTorch) untuk klasifikasi pose dari gambar menggunakan landmark MediaPipe."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_split"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models/lstm_pose"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_names(train_dir: Path) -> list[str]:
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"Tidak ada folder kelas di {train_dir}")
    return classes


def extract_landmark_sequence(image_path: Path, pose_detector) -> np.ndarray | None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(image_rgb)
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark
    seq = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)  # (33,4)

    # Normalisasi agar model fokus ke gerakan, bukan posisi absolut di kamera.
    hip_center = (seq[23, :3] + seq[24, :3]) / 2.0
    shoulder_width = np.linalg.norm(seq[11, :3] - seq[12, :3])
    scale = float(max(shoulder_width, 1e-6))
    seq[:, :3] = (seq[:, :3] - hip_center) / scale
    return seq


def load_split(
    split_dir: Path, class_to_idx: dict[str, int], pose_detector
) -> tuple[np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    y_data: list[int] = []

    for class_name, class_idx in class_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        image_paths = sorted([*class_dir.glob("*.jpg"), *class_dir.glob("*.jpeg"), *class_dir.glob("*.png")])
        for image_path in image_paths:
            seq = extract_landmark_sequence(image_path, pose_detector)
            if seq is None:
                continue
            x_data.append(seq)
            y_data.append(class_idx)

    if not x_data:
        raise ValueError(f"Tidak ada sample valid di split {split_dir}")

    x = np.stack(x_data).astype(np.float32)
    y = np.array(y_data, dtype=np.int64)
    return x, y


class PoseLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = torch.argmax(logits, dim=1)
    return (y_pred == y_true).float().mean().item()


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: list[int] = []
    all_true: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            y_pred = torch.argmax(logits, dim=1)
            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (y_pred == y_batch).sum().item()
            total_samples += batch_size

            all_preds.extend(y_pred.cpu().numpy().tolist())
            all_true.extend(y_batch.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc, np.array(all_true), np.array(all_preds)


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    test_dir = args.data_dir / "test"

    class_names = get_class_names(train_dir)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print("Class mapping:", class_to_idx)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose_detector:
        x_train, y_train = load_split(train_dir, class_to_idx, pose_detector)
        x_val, y_val = load_split(val_dir, class_to_idx, pose_detector)
        x_test, y_test = load_split(test_dir, class_to_idx, pose_detector)

    print(
        "Loaded samples:",
        f"train={len(x_train)}",
        f"val={len(x_val)}",
        f"test={len(x_test)}",
    )

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseLSTM(input_size=4, hidden_size=96, num_layers=2, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = None
    best_model_path = args.output_dir / "best_model.pth"
    patience = 8
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            y_pred = torch.argmax(logits, dim=1)
            epoch_loss += loss.item() * batch_size
            epoch_correct += (y_pred == y_batch).sum().item()
            total_samples += batch_size

        train_loss = epoch_loss / max(total_samples, 1)
        train_acc = epoch_correct / max(total_samples, 1)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, best_model_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping: val_acc tidak membaik selama {patience} epoch.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\n=== TEST METRICS ===")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)

    save_confusion_matrix(y_true, y_pred, class_names, args.output_dir / "confusion_matrix_test.png")

    metrics_json = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "class_names": class_names,
        "num_samples": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "best_model_path": str(best_model_path.resolve()),
        "history": {k: [float(vv) for vv in v] for k, v in history.items()},
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    with (args.output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nModel tersimpan di: {best_model_path.resolve()}")
    print(f"Metrics JSON       : {(args.output_dir / 'metrics.json').resolve()}")
    print(
        f"Confusion matrix   : {(args.output_dir / 'confusion_matrix_test.png').resolve()}"
    )


if __name__ == "__main__":
    main()
