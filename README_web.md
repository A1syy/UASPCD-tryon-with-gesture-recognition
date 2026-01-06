# Webcam Filter Web App — README

Aplikasi web untuk menerapkan face filter secara real-time menggunakan webcam. File ini menjelaskan penggunaan dan fitur dari script `web_app.py`.

## Ringkasan

Versi web-based dari aplikasi face filter dengan tampilan modern yang mengikuti **10 Nielsen's Usability Heuristics**:

1. **Visibility of system status** - Status kamera (Live/Paused), FPS, gesture terakhir ditampilkan real-time
2. **Match between system and real world** - Ikon dan label intuitif (🎭, ✋, ✌️, 👆)
3. **User control and freedom** - Tombol Reset, Pause, Clear Mask tersedia
4. **Consistency and standards** - Desain konsisten dengan komponen yang familiar
5. **Error prevention** - Validasi dan feedback langsung pada setiap aksi
6. **Recognition rather than recall** - Visual mask selector dengan thumbnail
7. **Flexibility and efficiency** - Keyboard shortcuts untuk power users
8. **Aesthetic and minimalist design** - UI bersih dengan warna yang harmonis
9. **Help users recognize errors** - Toast notification untuk error handling
10. **Help and documentation** - Modal bantuan dengan pintasan keyboard lengkap

## Fitur

- 🎬 **Live Preview** - Video stream real-time dengan filter
- 🎭 **Mask Selector** - Grid visual dengan thumbnail untuk memilih mask
- 🎚️ **Parameter Controls** - Slider untuk scale, posisi, rotasi
- 🖐️ **Gesture Control** - Kontrol menggunakan gerakan tangan
- ⌨️ **Keyboard Shortcuts** - Pintasan keyboard untuk efisiensi
- 📱 **Responsive Design** - Tampilan optimal di berbagai ukuran layar

## Prasyarat

- Python 3.9+ (disarankan 64-bit)

### Setup (disarankan: virtualenv `.venv`)

Buat environment lokal:

```bash
py -m venv .venv
```

Install dependencies dari `requirements.txt`:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Menjalankan

Pastikan folder berisi gambar mask tersedia (misal `mask/`). Jalankan:

```bash
.\.venv\Scripts\python.exe web_app.py --masks-folder mask
```

Kemudian buka browser dan akses:
```
http://127.0.0.1:5000
```

### Opsi Command Line

```bash
.\.venv\Scripts\python.exe web_app.py --masks-folder mask --port 5000 --host 127.0.0.1
```

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--masks-folder` | `mask` | Path ke folder mask |
| `--port` | `5000` | Port server |
| `--host` | `127.0.0.1` | Host server |

## Pintasan Keyboard

| Tombol | Fungsi |
|--------|--------|
| `Spasi` | Pause / Resume |
| `R` | Reset semua parameter |
| `C` | Hapus mask (clear) |
| `←` | Mask sebelumnya |
| `→` | Mask selanjutnya |
| `G` | Toggle kontrol gesture |
| `I` | Toggle info overlay |
| `H` | Tampilkan bantuan |
| `Esc` | Tutup dialog |

## Gesture Tangan

Aktifkan toggle "Kontrol Gesture" di panel atau tekan `G`:

| Gesture | Aksi |
|---------|------|
| ✋ Palm (telapak terbuka) | Hapus mask |
| ✌️ V Sign | Mask selanjutnya |
| 👆 Index pointing | Mask sebelumnya |

## API Endpoints

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/` | GET | Halaman utama |
| `/video_feed` | GET | Video stream (MJPEG) |
| `/api/masks` | GET | Daftar mask tersedia |
| `/api/mask` | POST | Set mask aktif |
| `/api/params` | GET/POST | Get/Set parameter |
| `/api/reset` | POST | Reset parameter |
| `/api/toggle` | POST | Toggle pengaturan |
| `/api/status` | GET | Status aplikasi |

## Struktur File

```
project/
├── web_app.py              # Backend Flask
├── templates/
│   └── index.html          # Template HTML
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet
│   └── js/
│       └── app.js          # JavaScript frontend
├── mask/                   # Folder mask images
│   ├── Ironman.png
│   ├── Spiderman.png
│   └── ...
└── filter_ref.py           # Filter engine
```

## Troubleshooting

- **Kamera tidak terdeteksi**: Pastikan tidak ada aplikasi lain yang menggunakan kamera
- **Video tidak muncul**: Refresh halaman atau periksa console browser
- **Gesture tidak terdeteksi**: Pastikan pencahayaan cukup dan tangan terlihat jelas
- **Port sudah digunakan**: Gunakan port lain dengan `--port 5001`
