# Webcam Local GUI — README

Aplikasi desktop untuk menerapkan face filter secara real-time menggunakan webcam. File ini menjelaskan penggunaan dan fitur dari script `webcam_local.py`.

## Ringkas

- Satu window: preview di kiri, panel kontrol di kanan (scrollable).
- Dropdown untuk memilih mask, slider khusus untuk parameter (scale, offset X/Y, yaw, pitch, roll).
- Scroll standar: mouse wheel, scrollbar track + thumb drag.
- Toggle: Pause, Show Info, Hand Control.
- Gesture tangan (MediaPipe): Open → Clear, V → Next, Index → Prev; skeleton tangan ditampilkan di preview.

## Prasyarat

- Python 3.9+ (disarankan 64-bit)
- Dependencies (lihat `requirements.txt`):
  - opencv-python
  - numpy
  - mediapipe

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Menjalankan

Pastikan folder berisi gambar mask tersedia (misal `mask/`). Jalankan:

```bash
python webcam_local.py --masks-folder mask
```

Opsional preload mask tertentu:

```bash
python webcam_local.py --masks-folder mask --mask bali.png
```

Jika `--masks-folder` tidak diberikan, aplikasi akan mencoba auto-detect `mask/` atau `masks/` di folder yang sama.

## Panel Kontrol (GUI)

- Current Mask: menampilkan mask aktif saat ini.
- Mask Select (Dropdown): klik untuk membuka daftar mask, klik item untuk memilih.
- Slider Controls (scrollable):
  - Scale %: ukuran mask (50–400).
  - Offset X / Offset Y: posisi horizontal/vertikal (−200 hingga +200).
  - Yaw % / Pitch %: intensitas rotasi (0–300).
  - Roll Deg: kemiringan (−180 hingga +180).
- Toggles:
  - Pause: hentikan pemrosesan sementara.
  - Show Info: tampilkan overlay status/FPS di preview.
  - Hand Control: aktifkan kontrol gestur tangan.
- Scrolling:
  - Mouse wheel di area panel untuk scroll seluruh panel.
  - Mouse wheel di area container slider hanya menggulir daftar slider.
  - Scrollbar di kanan: klik track, drag thumb.

## Gesture Tangan (Hand Control)

Aktifkan toggle "Hand Control" di panel.

- Skeleton tangan otomatis muncul di preview ketika tangan terdeteksi.
- Gestur yang didukung:
  - Palm (telapak terbuka, empat jari terlihat) → Clear Mask
  - V Sign (index + middle; ring + little terlipat) → Next Mask
  - Index pointing (hanya jari telunjuk) → Previous Mask
- Cooldown: ada jeda ~0.8 detik antar aksi untuk mencegah trigger berulang.
- Tips deteksi:
  - Pastikan pencahayaan cukup dan tangan menghadap kamera.
  - Tahan gestur selama ±0.5 detik agar stabil.

## Overlay Preview

- Sudut kiri atas menampilkan FPS dan status (LIVE/PAUSED).
- Saat Hand Control aktif, muncul legenda ringkas:
  - Palm → Clear
  - V → Next
  - Index → Prev
- Pesan terakhir gestur juga ditampilkan (mis. "Gesture: NEXT (V)").

## Folder Mask

- `--masks-folder` harus berisi file gambar (PNG/JPG) untuk mask.
- Nama file digunakan sebagai pilihan di dropdown; `[No Mask]` tersedia untuk menghapus mask.

## Troubleshooting

- Wheel tidak menggulir: pastikan window fokus pada "Webcam Filter Preview"; coba gunakan mouse wheel di area panel (kanan). Pada sebagian touchpad, scroll bisa lebih halus; aplikasi tetap merespons bertahap.
- Kamera gagal dibuka: pastikan tidak dipakai program lain; coba `cv2.CAP_DSHOW` sudah disetel; gunakan webcam internal/USB yang didukung.
- MediaPipe gagal: pastikan `mediapipe` terinstal; jika tidak tersedia, fitur gestur otomatis dinonaktifkan.
- Kinerja rendah: kurangi resolusi kamera, tutup aplikasi lain, pastikan GPU/CPU tidak overutilized.

## Pintasan Keyboard

- ESC: keluar.
- R: reset seluruh parameter ke default.

## Kustomisasi

- Cooldown gestur: ubah nilai `self.gesture_cooldown` di `webcam_local.py`.
- Default parameter slider: lihat bagian `reset_parameters()`.
- Lebar panel/preview: lihat `self.preview_width`, `self.preview_height`, dan `self.control_width`.

## Lisensi & Kredit

- Menggunakan OpenCV dan MediaPipe (© Google) untuk deteksi tangan.
- Script ini dibuat untuk keperluan GUI lokal face filter dan dapat disesuaikan.
