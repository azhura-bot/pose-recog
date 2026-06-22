# Pose Runner

`Pose Runner` sekarang punya 2 mode:

- `Mode online`: full client-side, siap deploy ke Vercel atau Netlify.
- `Mode legacy`: frontend HTML + backend Python lokal untuk eksperimen lama.

## Mode online

Mode ini adalah jalur paling realistis kalau project ingin dimainkan orang lain lewat link.

Karakteristik:

- Game runner berjalan di browser lewat `canvas`
- Pose detection memakai `MediaPipe Tasks Vision` di browser
- Kamera diproses di perangkat user, bukan di server
- Tidak perlu `Flask`, `pyautogui`, atau `127.0.0.1`

File utama:

- `index.html`
- `game.html`
- `game-online.js`
- `styles.css`
- `script.js`

### Deploy ke Vercel

1. Push repo ke GitHub.
2. Import repo ke Vercel.
3. Framework preset pilih `Other`.
4. Tidak perlu build command.
5. Output directory kosongkan atau isi `.`
6. Deploy.

Catatan:

- File `.vercelignore` sudah dibuat agar dataset, model, dan file Python tidak ikut ter-upload.
- Camera API butuh `https`, dan Vercel sudah memenuhi itu.

### Deploy ke Netlify

1. Push repo ke GitHub.
2. Import repo ke Netlify.
3. Build command kosong.
4. Publish directory isi `.`
5. Deploy.

Catatan:

- `netlify.toml` sudah disiapkan untuk publish root statis.
- File `.netlifyignore` mengecualikan aset training yang tidak dibutuhkan online.

### Mapping pose online

- Tangan kiri naik -> `kiri`
- Tangan kanan naik -> `kanan`
- Kedua tangan naik -> `lompat`
- Jongkok -> `slide`

Fallback keyboard:

- `ArrowLeft`
- `ArrowRight`
- `ArrowUp`
- `ArrowDown`

## Mode legacy Python

Mode lama masih relevan kalau kamu memang ingin eksperimen lokal dengan:

- `Flask`
- stream `video_feed`
- kontrol keyboard berbasis `pyautogui` / `pydirectinput`

Setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/game`

Catatan penting:

- Mode legacy tidak cocok untuk deploy ke Vercel/Netlify.
- Masalah utamanya adalah kontrol game dilakukan dari backend lokal, bukan dari browser user.
