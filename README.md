# 🎭 Face Mood Counter — Real-time Emotion Analytics

> Deteksi emosi wajah secara real-time dari webcam, hitung jumlah orang per emosi, dan kirim hasilnya ke **Google Sheets** secara langsung — **tanpa TensorFlow**, ringan, dan mudah dipasang.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16%2B-orange?logo=onnx&logoColor=white)
![Google Apps Script](https://img.shields.io/badge/Google%20Apps%20Script-Webhook-34A853?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Tentang Proyek

Aplikasi ini mendeteksi ekspresi wajah dari kamera secara real-time menggunakan **OpenCV** untuk menangkap & mendeteksi wajah, dan model **ONNX `emotion-ferplus-8`** untuk mengklasifikasi 8 emosi. Setiap orang yang terdeteksi dihitung **satu kali** (lewat *centroid tracking*), lalu rekapnya dikirim ke **Google Sheets** melalui webhook Google Apps Script — cocok untuk analitik suasana ruang kelas, acara, atau area publik.

Dibuat sepenuhnya di atas **ONNX Runtime** sehingga tidak bergantung pada TensorFlow yang sering bermasalah saat instalasi.

---

## ✨ Fitur

- 🎥 **Deteksi emosi real-time** dari webcam (8 kelas emosi).
- 👥 **Penghitung orang per emosi** dengan *centroid tracking* — satu orang dihitung sekali, bukan per frame.
- 📊 **Integrasi Google Sheets** real-time via webhook Apps Script (tab `Log` + `Summary` otomatis).
- 🪶 **Ringan & bebas TensorFlow** — cukup `opencv-python`, `onnxruntime`, `numpy`.
- 🎚️ **Penyetelan langsung** sensitivitas emosi saat program berjalan (tombol `[` / `]`).
- 🖼️ **Panel statistik terpisah** di samping video, tidak menutupi wajah.
- 💾 **Cadangan lokal CSV** sehingga data aman walau webhook gagal.
- 🔄 **Reset satu tombol** (`r`) untuk panel lokal sekaligus tally di Google Sheets.
- 🇮🇩 **Label Bahasa Indonesia** (Netral, Senang, Terkejut, Sedih, Marah, Jijik, Takut, Sinis).

---

## 🧠 Cara Kerja

```
┌────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────┐
│  Webcam    │ → │ OpenCV Haar       │ → │ ONNX emotion-       │ → │ Centroid     │
│  (OpenCV)  │   │ Cascade (wajah)   │   │ ferplus-8 (emosi)  │   │ Tracking (ID)│
└────────────┘   └──────────────────┘   └────────────────────┘   └──────┬───────┘
                                                                         │ 1 orang = 1 hitung
                                                                         ▼
                                              ┌──────────────────────────────────┐
                                              │ Webhook → Google Apps Script      │
                                              │ → Google Sheets (Log + Summary)   │
                                              └──────────────────────────────────┘
```

1. Wajah dideteksi tiap frame, di-crop persegi + *histogram equalization* agar emosi lebih terbaca.
2. Model ONNX mengeluarkan probabilitas 8 emosi (dengan kompensasi bias "Netral").
3. Tiap wajah diberi ID; setelah stabil beberapa frame, orang itu dihitung sekali dengan emosi dominan.
4. Event dikirim ke Google Sheets dan dicatat ke CSV lokal.

---

## 🗂️ Struktur File

| File | Keterangan |
|------|------------|
| `face_mood_counter_gsheet.py` | **Aplikasi utama** — counter + integrasi Google Sheets |
| `face_mood_detection_onnx.py` | Versi standalone deteksi emosi (tanpa counter & Sheets) |
| `AppsScript_Code.gs` | Kode Google Apps Script (webhook penerima data) |
| `requirements_onnx.txt` | Daftar dependensi Python |
| `models/emotion-ferplus-8.onnx` | Model emosi (di-download otomatis saat pertama jalan) |

---

## 🚀 Instalasi

### Prasyarat
- Python **3.10 – 3.13** (64-bit)
- Webcam
- Koneksi internet (sekali, untuk unduh model ~35 MB)

### Langkah

```bash
# 1. Clone repository
git clone https://github.com/DESR217/face-mood-counter.git
cd face-mood-counter

# 2. (Disarankan) buat virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependensi
pip install -r requirements_onnx.txt

# 4. Jalankan
python face_mood_counter_gsheet.py
```

> 💡 **Di PyCharm:** buat project baru dengan interpreter Python 3.10–3.13, masukkan file ke folder project, lalu `pip install -r requirements_onnx.txt` di Terminal dan klik **Run**.

---

## 📑 Setup Google Sheets (opsional, untuk fitur counter online)

1. Buka **[sheets.new](https://sheets.new)** untuk membuat Google Sheet baru (tab `Log` & `Summary` akan dibuat otomatis).
2. Di sheet itu: menu **Extensions → Apps Script**.
3. Hapus isi `Code.gs`, tempel seluruh isi **`AppsScript_Code.gs`**, lalu **Save**.
4. **Deploy → New deployment → Web app**:
   - *Execute as:* **Me**
   - *Who has access:* **Anyone**
5. Salin **Web app URL**, lalu tempel ke variabel `APPS_SCRIPT_URL` di `face_mood_counter_gsheet.py`:

```python
APPS_SCRIPT_URL = "https://script.google.com/macros/s/XXXX/exec"
```

> 🔒 **Catatan keamanan:** akses *Anyone* diperlukan agar webhook bisa menerima data tanpa login. Jangan sebar URL-nya. Untuk produksi, tambahkan token rahasia di kedua sisi.

Jika `APPS_SCRIPT_URL` dibiarkan kosong, aplikasi tetap berjalan dalam **mode lokal** (menghitung + menyimpan ke CSV).

---

## ⌨️ Kontrol Keyboard

| Tombol | Fungsi |
|:------:|--------|
| `q` | Keluar dari aplikasi |
| `s` | Simpan screenshot |
| `r` | Reset counter (panel lokal **+** tab Summary di Sheets) |
| `[` | Turunkan kompensasi Netral (emosi lain lebih mudah muncul) |
| `]` | Naikkan kompensasi Netral |

---

## ⚙️ Konfigurasi

Atur di bagian atas `face_mood_counter_gsheet.py`:

| Parameter | Default | Keterangan |
|-----------|:-------:|------------|
| `CAM_INDEX` | `0` | Indeks webcam (ganti 1/2 bila tidak terdeteksi) |
| `APPS_SCRIPT_URL` | `""` | URL Web App Apps Script (kosong = mode lokal) |
| `CONFIRM_FRAMES` | `10` | Jumlah frame stabil sebelum 1 orang dihitung |
| `NEUTRAL_PENALTY` | `0.50` | Penekan bias "Netral" (1.0 = nonaktif) |
| `FACE_MARGIN` | `0.25` | Margin crop wajah |
| `MIN_FACE` | `80` | Ukuran wajah minimum (px) |
| `MAX_DISAPPEARED` | `40` | Toleransi frame wajah hilang sesaat |
| `MAX_DISTANCE` | `140` | Jarak px maksimum mencocokkan wajah antar frame |

---

## 😀 Kelas Emosi

| Model (EN) | Label (ID) |
|------------|------------|
| neutral | Netral |
| happiness | Senang |
| surprise | Terkejut |
| sadness | Sedih |
| anger | Marah |
| disgust | Jijik |
| fear | Takut |
| contempt | Sinis |

Model: [`emotion-ferplus-8`](https://huggingface.co/onnxmodelzoo/emotion-ferplus-8) (ONNX Model Zoo), input grayscale 64×64, ±35 MB.

---

## ⚠️ Keterbatasan

- **Bias Netral:** model FER+ dilatih di FER2013 dan cenderung memilih "Netral". Sudah dikompensasi (crop persegi, *histogram equalization*, penalti Netral), tapi tidak sempurna untuk ekspresi halus.
- **Deteksi wajah:** Haar cascade kurang akurat pada wajah miring / pencahayaan buruk.
- **Tracking berbasis posisi, bukan pengenalan wajah:** orang yang **keluar lalu masuk lagi** ke frame akan dihitung sebagai orang baru. Dua wajah yang berpapasan & saling menutupi bisa tertukar ID.

---

## 🗺️ Roadmap

- [ ] Upgrade deteksi wajah ke **YuNet** (`cv2.FaceDetectorYN`) untuk akurasi lebih baik.
- [ ] **Face recognition** (embedding ONNX) agar orang yang sama tak terhitung dobel sepanjang waktu.
- [ ] Token rahasia untuk mengamankan webhook.
- [ ] Dashboard visual (grafik tren emosi dari data Sheets).

---

## 🛠️ Tech Stack

`Python` · `OpenCV` · `ONNX Runtime` · `NumPy` · `Google Apps Script` · `Google Sheets`

---

## 📄 Lisensi

Dirilis di bawah lisensi **MIT**. Bebas digunakan, dimodifikasi, dan didistribusikan.

---

## 🏷️ GitHub Topics

Tambahkan topics berikut di repo (Settings → ⚙️ di samping *About* → *Topics*):

```
computer-vision  opencv  onnx  onnxruntime  emotion-detection
emotion-recognition  facial-expression-recognition  face-detection
mood-detection  people-counter  centroid-tracking  real-time
python  google-sheets  google-apps-script  webhook  ferplus
deep-learning  machine-learning  analytics
```

## #️⃣ Hashtag (untuk promosi LinkedIn / X)

```
#ComputerVision #OpenCV #ONNX #Python #EmotionDetection #EmotionRecognition
#MachineLearning #DeepLearning #ArtificialIntelligence #AI #FaceDetection
#RealTime #GoogleSheets #AppsScript #DataAnalytics #MoodDetection
#PeopleCounter #MLOps #OpenSource #TechIndonesia
```

---

<p align="center">Dibuat dengan ❤️ menggunakan OpenCV & ONNX Runtime</p>
