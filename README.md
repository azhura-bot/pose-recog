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
