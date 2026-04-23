# Pose Runner (HTML/CSS/JS + Python Backend)

Project ini berisi:
- Landing page frontend (`index.html`)
- Halaman game 2 panel (`game.html`):
  - Kiri: iframe game
  - Kanan: stream kamera real-time dengan landmark pose (MediaPipe)
- Backend Python (`app.py`) untuk pose recognition dan kontrol keyboard (`pyautogui`)

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Jalankan

```bash
python app.py
```

Buka:
- `http://127.0.0.1:5000/` (landing page)
- `http://127.0.0.1:5000/game` (halaman game + stream pose)

Saat startup pertama, backend akan mengunduh model `pose_landmarker_full.task` ke folder `models/`.

## Pose Mapping (default)

- Geser badan ke kiri -> `Left`
- Geser badan ke kanan -> `Right`
- Jongkok -> `Down` (slide)
- Lompat -> `Up` (jump)
- Tepuk tangan (kedua tangan saling mendekat di depan badan) -> `Esc` (pause/resume)

Catatan:
- Fokuskan jendela browser/game agar input keyboard dari `pyautogui` masuk ke game.
- Jika `pyautogui` belum ter-install atau gagal di environment tertentu, landmark tetap berjalan tapi kontrol keyboard nonaktif.

## Training Deep Learning (LSTM)

Pipeline training ini menggunakan:
- Landmark pose dari MediaPipe (tiap gambar jadi urutan 33 titik)
- Model `LSTM` berbasis PyTorch
- Split data `train/val/test` secara stratified per kelas

### 1) Split dataset

```bash
python split_dataset.py --input-dir dataset --output-dir dataset_split --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42 --overwrite
```

### 2) Training + evaluasi

```bash
python train_lstm_pose.py --data-dir dataset_split --epochs 30 --batch-size 16 --learning-rate 0.001 --output-dir models/lstm_pose
```

Output training tersimpan di `models/lstm_pose/`:
- `best_model.pth`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix_test.png`
