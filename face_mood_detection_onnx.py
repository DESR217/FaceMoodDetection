"""
Face Mood Detection (Real-time) — OpenCV + ONNX Runtime (TANPA TensorFlow)
==========================================================================
Versi diperbaiki: akurasi emosi lebih baik, tidak mudah mentok di "Netral".

Perbaikan dibanding versi awal:
  1. Crop wajah dibuat PERSEGI + margin  -> proporsi tidak gepeng saat resize 64x64
  2. Histogram equalization              -> normalisasi pencahayaan/kontras
  3. Kompensasi bias Netral (tunable)    -> ekspresi asli lebih mudah muncul

Pipeline:
    OpenCV webcam -> Haar cascade (deteksi wajah) -> ONNX emotion-ferplus-8 -> visual

Dependensi: opencv-python, onnxruntime, numpy
Model (35 MB) di-download otomatis sekali ke ./models/ saat pertama dijalankan.

Kontrol: q=keluar  s=screenshot  m=toggle smoothing
         [ / ]  -> turunkan / naikkan kompensasi Netral secara live
Author: dibuat untuk Wiko
"""

import hashlib
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Konfigurasi  (silakan diatur)
# ---------------------------------------------------------------------------
CAM_INDEX = 0
SMOOTH_WINDOW = 8        # rata-rata berapa frame (anti-kedip)

FACE_MARGIN = 0.25       # ruang ekstra di sekeliling wajah (0.25 = +25%)
USE_HIST_EQ = True       # True = normalisasi kontras (disarankan ON)
NEUTRAL_PENALTY = 0.55   # 1.0 = tanpa kompensasi. Makin kecil = Netral makin ditekan.
                         # Rekomendasi 0.4-0.7. Bisa diubah live dengan tombol [ dan ]
MIN_FACE = 80            # ukuran minimum wajah (px) — naikkan kalau banyak deteksi palsu

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")
MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/"
    "resolve/main/emotion-ferplus-8.onnx"
)
MODEL_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"

# index 0..7 -> (nama, label Indonesia, warna BGR).  index 0 = neutral.
EMOTIONS = [
    ("neutral",   "Netral",   (190, 190, 190)),
    ("happiness", "Senang",   (0, 220, 0)),
    ("surprise",  "Terkejut", (0, 215, 255)),
    ("sadness",   "Sedih",    (220, 120, 0)),
    ("anger",     "Marah",    (0, 0, 255)),
    ("disgust",   "Jijik",    (0, 153, 76)),
    ("fear",      "Takut",    (153, 0, 153)),
    ("contempt",  "Sinis",    (128, 128, 0)),
]


# ---------------------------------------------------------------------------
# Download model (sekali saja)
# ---------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[INFO] Mengunduh model (~35 MB) dari Hugging Face ...")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(MODEL_PATH, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"Gagal download model: {e}\n"
            f"Unduh manual dari:\n  {MODEL_URL}\n"
            f"simpan sebagai:\n  {os.path.abspath(MODEL_PATH)}"
        )
    sha = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
    if sha != MODEL_SHA256:
        print(f"[WARN] SHA256 tidak cocok ({sha[:12]}...). Biasanya tetap jalan.")
    print(f"[INFO] Model tersimpan: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Pemrosesan wajah & inferensi
# ---------------------------------------------------------------------------
def crop_face_square(gray, x, y, w, h, margin):
    """Ambil wajah sebagai area PERSEGI dengan margin, supaya tidak terdistorsi."""
    H, W = gray.shape
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin)
    x1 = int(max(cx - side / 2, 0)); y1 = int(max(cy - side / 2, 0))
    x2 = int(min(cx + side / 2, W)); y2 = int(min(cy + side / 2, H))
    return gray[y1:y2, x1:x2]


def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def predict_emotion(session, input_name, face_gray):
    if face_gray.size == 0:
        return None
    if USE_HIST_EQ:
        face_gray = cv2.equalizeHist(face_gray)
        # Alternatif lebih adaptif (kadang lebih bagus untuk pencahayaan tidak rata):
        # face_gray = cv2.createCLAHE(2.0, (8, 8)).apply(face_gray)
    img = cv2.resize(face_gray, (64, 64)).astype(np.float32)
    blob = img.reshape(1, 1, 64, 64)
    scores = session.run(None, {input_name: blob})[0].flatten()
    return softmax(scores)


def apply_neutral_penalty(probs, penalty):
    """Tekan probabilitas Netral (index 0) lalu normalisasi ulang."""
    if penalty >= 1.0:
        return probs
    p = probs.copy()
    p[0] *= penalty
    return p / p.sum()


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------
def draw_label(frame, x, y, w, h, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_emotion_bars(frame, probs, origin=(10, 40)):
    x0, y0 = origin; bar_w_max = 180; gap = 26
    for i, p in enumerate(probs):
        _, label_id, color = EMOTIONS[i]
        y = y0 + i * gap
        cv2.rectangle(frame, (x0 + 90, y - 12), (x0 + 90 + int(p * bar_w_max), y + 4),
                      color, -1)
        cv2.rectangle(frame, (x0 + 90, y - 12), (x0 + 90 + bar_w_max, y + 4),
                      (80, 80, 80), 1)
        cv2.putText(frame, label_id, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{p*100:4.0f}%", (x0 + 90 + bar_w_max + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global NEUTRAL_PENALTY
    ensure_model()

    print("[INFO] Memuat model ONNX ...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam index {CAM_INDEX} tidak bisa dibuka.")

    smoothing = True
    buf = deque(maxlen=SMOOTH_WINDOW)
    prev_t = time.time(); fps = 0.0
    print("[INFO] Mulai. q=keluar s=screenshot m=smoothing  [ ]=atur kompensasi Netral")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE, MIN_FACE))

        main_probs = None; main_area = 0
        for (x, y, w, h) in faces:
            roi = crop_face_square(gray, x, y, w, h, FACE_MARGIN)
            probs = predict_emotion(session, input_name, roi)
            if probs is None:
                continue
            probs = apply_neutral_penalty(probs, NEUTRAL_PENALTY)

            top = int(np.argmax(probs))
            _, label_id, color = EMOTIONS[top]
            draw_label(frame, x, y, w, h, f"{label_id} {probs[top]*100:.0f}%", color)
            if w * h > main_area:
                main_area = w * h; main_probs = probs

        if main_probs is not None:
            if smoothing:
                buf.append(main_probs)
                draw_emotion_bars(frame, np.mean(np.array(buf), axis=0))
            else:
                draw_emotion_bars(frame, main_probs)
        else:
            cv2.putText(frame, "Wajah tidak terdeteksi", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            buf.clear()

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6)); prev_t = now
        cv2.putText(frame,
                    f"FPS:{fps:.1f}  smooth:{'ON' if smoothing else 'OFF'}  "
                    f"neutralPenalty:{NEUTRAL_PENALTY:.2f}",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Face Mood Detection - OpenCV + ONNX", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"mood_{int(time.time())}.png"; cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot: {fname}")
        elif key == ord("m"):
            smoothing = not smoothing; buf.clear()
        elif key == ord("["):
            NEUTRAL_PENALTY = max(0.1, round(NEUTRAL_PENALTY - 0.05, 2))
        elif key == ord("]"):
            NEUTRAL_PENALTY = min(1.0, round(NEUTRAL_PENALTY + 0.05, 2))

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()