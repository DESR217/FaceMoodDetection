"""
Face Mood Counter -> Google Sheets (Real-time) — OpenCV + ONNX
==============================================================
Menghitung BERAPA ORANG (bukan berapa frame) per emosi, dikirim ke Google
Sheets via Apps Script Web App.

Perbaikan versi ini:
  - Panel counter kecil & SEMI-TRANSPARAN, tidak menutupi wajah
  - Deteksi mood = logika standalone (crop persegi + histeq + penalti Netral),
    penalti bisa diatur LIVE dengan [ dan ]
  - Tracking lebih tahan banting -> orang yang sama tidak gampang dihitung ulang
    (toleransi kehilangan wajah dinaikkan)

Dependensi: opencv-python, onnxruntime, numpy   (webhook pakai urllib stdlib)

Kontrol: q=keluar  s=screenshot  r=reset hitungan  [ ]=atur penalti Netral
Author: dibuat untuk Wiko
"""

import csv
import hashlib
import json
import os
import queue
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
CAM_INDEX = 0

# >>> ISI dengan URL Web App Apps Script (lihat AppsScript_Code.gs).
# Kosongkan ("") untuk jalan tanpa Sheets (tetap hitung + simpan CSV lokal).
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzDv_85klYz7Np5NiqAldBFeCk41YGdokIO2X2hRgZa3ZtEevnzyrVO-aXAUl3FVSE/exec"

CONFIRM_FRAMES = 10      # frame stabil sebelum 1 orang dihitung (cukup utk tangkap ekspresi)
FACE_MARGIN = 0.25
USE_HIST_EQ = True
NEUTRAL_PENALTY = 0.50   # 1.0 = tanpa kompensasi; makin kecil = Netral makin ditekan
MIN_FACE = 80

# Tracking — dinaikkan supaya wajah yang sama tidak gampang jadi "orang baru"
MAX_DISAPPEARED = 40     # frame toleransi wajah hilang sesaat (kedip/menoleh)
MAX_DISTANCE = 140       # jarak px maksimum mencocokkan wajah antar frame

LOCAL_CSV = "mood_counter_log.csv"
WIB = timezone(timedelta(hours=7))

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")
MODEL_URL = ("https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/"
             "resolve/main/emotion-ferplus-8.onnx")
MODEL_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"

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
# Kirim ke Sheets (thread terpisah, biar webcam tidak nge-lag)
# ---------------------------------------------------------------------------
event_q = queue.Queue()


def webhook_worker():
    while True:
        payload = event_q.get()
        if payload is None:
            event_q.task_done()
            break
        if APPS_SCRIPT_URL:
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    APPS_SCRIPT_URL, data=data,
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10)
            except Exception as e:
                print(f"[WARN] Gagal kirim ke Sheets: {e}")
        event_q.task_done()


def log_event(payload):
    new = not os.path.exists(LOCAL_CSV)
    with open(LOCAL_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["Waktu (WIB)", "Person ID", "Emotion",
                        "Emosi", "Confidence", "Frames"])
        w.writerow([payload["timestamp"], payload["person_id"],
                    payload["emotion_en"], payload["emotion_id"],
                    payload["confidence"], payload["frames"]])
    event_q.put(payload)


# ---------------------------------------------------------------------------
# Centroid Tracker
# ---------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared, max_distance):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, c):
        oid = self.next_id
        self.objects[oid] = c
        self.disappeared[oid] = 0
        self.next_id += 1
        return oid

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {}

        centroids = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]
        assign = {}

        if len(self.objects) == 0:
            for i, c in enumerate(centroids):
                assign[self.register(c)] = i
        else:
            ids = list(self.objects.keys())
            obj_c = np.array(list(self.objects.values()))
            inp_c = np.array(centroids)
            D = np.linalg.norm(obj_c[:, None] - inp_c[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_r, used_c = set(), set()
            for r, c in zip(rows, cols):
                if r in used_r or c in used_c or D[r, c] > self.max_distance:
                    continue
                oid = ids[r]
                self.objects[oid] = centroids[c]
                self.disappeared[oid] = 0
                assign[oid] = c
                used_r.add(r); used_c.add(c)
            for r in set(range(D.shape[0])) - used_r:
                oid = ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            for c in set(range(D.shape[1])) - used_c:
                assign[self.register(centroids[c])] = c

        return {oid: rects[i] for oid, i in assign.items()}


# ---------------------------------------------------------------------------
# Model & inferensi  (logika sama dengan versi standalone yang diperbaiki)
# ---------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[INFO] Mengunduh model (~35 MB) ...")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(MODEL_PATH, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        raise RuntimeError(f"Gagal download model: {e}\nUnduh manual: {MODEL_URL}")
    if hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest() != MODEL_SHA256:
        print("[WARN] SHA256 tidak cocok, biasanya tetap jalan.")


def crop_face_square(gray, x, y, w, h, margin):
    H, W = gray.shape
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + margin)
    x1 = int(max(cx - side / 2, 0)); y1 = int(max(cy - side / 2, 0))
    x2 = int(min(cx + side / 2, W)); y2 = int(min(cy + side / 2, H))
    return gray[y1:y2, x1:x2]


def softmax(v):
    e = np.exp(v - np.max(v)); return e / e.sum()


def predict_emotion(session, in_name, face_gray, penalty):
    if face_gray.size == 0:
        return None
    if USE_HIST_EQ:
        face_gray = cv2.equalizeHist(face_gray)
    img = cv2.resize(face_gray, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
    probs = softmax(session.run(None, {in_name: img})[0].flatten())
    if penalty < 1.0:
        probs = probs.copy(); probs[0] *= penalty; probs /= probs.sum()
    return probs


# ---------------------------------------------------------------------------
# Panel counter semi-transparan (tidak menutupi wajah)
# ---------------------------------------------------------------------------
PANEL_W = 230  # lebar panel samping (px)


def render_panel(height, total, counts, penalty, fps):
    """Panel counter TERPISAH di samping video — sama sekali tidak menimpa wajah."""
    F = cv2.FONT_HERSHEY_SIMPLEX
    panel = np.full((height, PANEL_W, 3), 28, dtype=np.uint8)
    cv2.line(panel, (PANEL_W - 1, 0), (PANEL_W - 1, height), (70, 70, 70), 1)

    cv2.putText(panel, "MOOD COUNTER", (14, 30), F, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"TOTAL: {total}", (14, 66), F, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for i, (en, idn, color) in enumerate(EMOTIONS):
        y = 104 + i * 32
        cv2.putText(panel, idn, (14, y), F, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(panel, str(counts.get(en, 0)), (PANEL_W - 55, y),
                    F, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"penalty {penalty:.2f}", (14, height - 38),
                F, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, f"fps {fps:.0f}", (14, height - 16),
                F, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global NEUTRAL_PENALTY
    ensure_model()
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam index {CAM_INDEX} tidak bisa dibuka.")

    threading.Thread(target=webhook_worker, daemon=True).start()
    if not APPS_SCRIPT_URL:
        print("[INFO] APPS_SCRIPT_URL kosong -> mode lokal (hitung + CSV saja).")

    tracker = CentroidTracker(MAX_DISAPPEARED, MAX_DISTANCE)
    tracks = {}
    counts = Counter()
    total_people = 0
    prev_t = time.time(); fps = 0.0
    print("[INFO] Mulai. q=keluar s=screenshot r=reset [ ]=atur penalti Netral")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE, MIN_FACE))

        objects = tracker.update([tuple(f) for f in faces])

        for oid, (x, y, w, h) in objects.items():
            probs = predict_emotion(session, in_name,
                                    crop_face_square(gray, x, y, w, h, FACE_MARGIN),
                                    NEUTRAL_PENALTY)
            if probs is None:
                continue
            t = tracks.setdefault(oid, {"votes": Counter(), "frames": 0,
                                        "counted": False})
            t["votes"][int(np.argmax(probs))] += 1
            t["frames"] += 1

            dom = t["votes"].most_common(1)[0][0]   # label live (selalu update)
            _, label_id, color = EMOTIONS[dom]

            if not t["counted"] and t["frames"] >= CONFIRM_FRAMES:
                t["counted"] = True
                total_people += 1
                en, idn, _ = EMOTIONS[dom]
                counts[en] += 1
                conf = round(t["votes"][dom] / t["frames"], 2)
                log_event({
                    "timestamp": datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S"),
                    "person_id": oid, "emotion_en": en, "emotion_id": idn,
                    "confidence": conf, "frames": t["frames"],
                })
                print(f"[HITUNG] Orang #{oid} -> {idn} ({conf}) | total {total_people}")

            mark = "OK" if t["counted"] else f"{t['frames']}/{CONFIRM_FRAMES}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"#{oid} {label_id} {mark}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        for oid in list(tracks.keys()):
            if oid not in tracker.objects:
                del tracks[oid]

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6)); prev_t = now
        panel = render_panel(frame.shape[0], total_people, counts, NEUTRAL_PENALTY, fps)
        canvas = np.hstack([panel, frame])   # panel di kiri, video di kanan (tidak menimpa)

        cv2.imshow("Face Mood Counter -> Google Sheets", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"counter_{int(time.time())}.png", frame)
        elif key == ord("r"):
            counts.clear(); total_people = 0; tracks.clear()
            print("[INFO] Hitungan sesi di-reset.")
        elif key == ord("["):
            NEUTRAL_PENALTY = max(0.1, round(NEUTRAL_PENALTY - 0.05, 2))
        elif key == ord("]"):
            NEUTRAL_PENALTY = min(1.0, round(NEUTRAL_PENALTY + 0.05, 2))

    cap.release()
    cv2.destroyAllWindows()
    event_q.put(None)
    print(f"[INFO] Selesai. Total orang terhitung: {total_people}")


if __name__ == "__main__":
    main()