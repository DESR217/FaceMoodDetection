/**
 * Mood Counter Webhook — Google Apps Script
 * =========================================
 * Menerima data dari skrip Python (face_mood_counter_gsheet.py) dan:
 *   - mencatat tiap orang ke sheet "Log"
 *   - memperbarui tally per emosi di sheet "Summary"
 *
 * CARA DEPLOY:
 *   1. Buka Google Sheet baru -> menu Extensions -> Apps Script
 *   2. Hapus isi Code.gs, tempel SELURUH kode ini, lalu Save
 *   3. Klik Deploy -> New deployment -> pilih tipe "Web app"
 *        - Execute as: Me
 *        - Who has access: Anyone
 *   4. Klik Deploy, salin "Web app URL"
 *   5. Tempel URL itu ke variabel APPS_SCRIPT_URL di skrip Python
 *
 * Catatan keamanan: URL ini bersifat publik (siapa pun yang punya URL bisa
 * mengirim data). Jangan disebar. Untuk produksi, tambahkan token rahasia.
 */

const TZ = "Asia/Jakarta";           // WIB (GMT+7)
const LOG_SHEET = "Log";
const SUMMARY_SHEET = "Summary";

// Urutan & label harus sama dengan di skrip Python
const EMOTIONS = [
  ["neutral", "Netral"], ["happiness", "Senang"], ["surprise", "Terkejut"],
  ["sadness", "Sedih"], ["anger", "Marah"], ["disgust", "Jijik"],
  ["fear", "Takut"], ["contempt", "Sinis"]
];

function doPost(e) {
  try {
    const b = JSON.parse(e.postData.contents);
    const ss = SpreadsheetApp.getActiveSpreadsheet();

    // Perintah reset dari tombol "r" di skrip Python -> nolkan tally Summary
    if (b.action === "reset") {
      resetSummary_(ss);
      return json_({ ok: true, reset: true });
    }

    const log = getOrCreateLog_(ss);
    const ts = Utilities.formatDate(new Date(), TZ, "yyyy-MM-dd HH:mm:ss");
    log.appendRow([ts, b.person_id, b.emotion_en, b.emotion_id,
                   b.confidence, b.frames]);

    bumpSummary_(ss, b.emotion_en, b.emotion_id);
    return json_({ ok: true });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  }
}

// Set semua angka "Jumlah" di sheet Summary kembali ke 0.
// Catatan: sheet "Log" sengaja TIDAK dihapus agar riwayat tetap tersimpan.
function resetSummary_(ss) {
  const s = ss.getSheetByName(SUMMARY_SHEET);
  if (!s) return;
  const last = s.getLastRow();
  if (last > 1) s.getRange(2, 3, last - 1, 1).setValue(0);
}

// Untuk tes di browser (buka Web app URL langsung)
function doGet() {
  return json_({ ok: true, msg: "Mood counter webhook aktif" });
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function getOrCreateLog_(ss) {
  let s = ss.getSheetByName(LOG_SHEET);
  if (!s) {
    s = ss.insertSheet(LOG_SHEET);
    s.appendRow(["Waktu (WIB)", "Person ID", "Emotion", "Emosi",
                 "Confidence", "Frames"]);
    s.getRange(1, 1, 1, 6).setFontWeight("bold");
  }
  return s;
}

function bumpSummary_(ss, en, idn) {
  let s = ss.getSheetByName(SUMMARY_SHEET);
  if (!s) {
    s = ss.insertSheet(SUMMARY_SHEET);
    s.appendRow(["Emotion", "Emosi", "Jumlah"]);
    s.getRange(1, 1, 1, 3).setFontWeight("bold");
    EMOTIONS.forEach(p => s.appendRow([p[0], p[1], 0]));
  }
  const last = s.getLastRow();
  const data = s.getRange(2, 1, last - 1, 3).getValues();
  for (let i = 0; i < data.length; i++) {
    if (data[i][0] === en) {
      s.getRange(i + 2, 3).setValue(Number(data[i][2]) + 1);
      return;
    }
  }
  s.appendRow([en, idn, 1]); // jaga-jaga kalau emosi belum ada
}
