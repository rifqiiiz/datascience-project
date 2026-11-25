## MINGGU 1 — [L] LEARNING PROBLEM

**Tujuan:** Mahasiswa memiliki ide sendiri yang selaras dengan satu SDG, dataset yang layak, dan rumusan Target-Objective-Data.

**Tugas:**

1. **Kembangkan ide sendiri berbasis SDGs** (referensi: https://sdgs.un.org/2030agenda):
   * Pelajari 17 SDGs dan targetnya, lalu pilih **satu** SDG utama yang paling dekat dengan minat/lingkungan Anda.
   * Identifikasi masalah nyata di bawah SDG tersebut (isu lokal, nasional, atau global).
   * Tentukan bagaimana masalah itu dapat dirumuskan sebagai tugas ML dan jenis datanya (tabular / image / text / time series / recommender).
   * Isi **Formulir Pengajuan Ide** (Lampiran B template) dan isi Bab 2.2 *Keterkaitan dengan SDGs*.
   * Daftar contoh di Lampiran A hanya inspirasi; ide wajib berasal dari mahasiswa.
2. Pilih dataset (Kaggle / UCI / lainnya). Pastikan ukurannya memadai untuk 3 model dan DL bisa dilatih di Colab/Kaggle.
3. Studi literatur: minimal 2 referensi ilmiah (format APA/IEEE).
4. Tulis latar belakang, 2–4 problem statements, dan goals terukur.
5. Isi tabel **Learning Problem** (Target, Objective, Data) di Bab 3.2.
6. Tentukan rencana tiga model (baseline, advanced, deep learning) dan alasan awalnya.
7. Buat repository GitHub kosong dan tambahkan dosen sebagai collaborator/pemilik akses baca.

**Deliverable:**
* Formulir Pengajuan Ide (Lampiran B)
* Proposal singkat 1–2 halaman (Bab 1–3 versi awal, termasuk Bab 2.2 keterkaitan SDGs)
* Link dataset dan repository GitHub

**Checklist selesai:**
* [x] Target dan jenis tugas ML jelas (klasifikasi/regresi/dll.)
* [x] Objective dan metrik utama ditentukan
* [x] Dataset dapat diakses dan sudah dicek ukurannya
* [x] Minimal 2 referensi terkumpul
* [x] Ide berasal dari mahasiswa dan selaras dengan satu SDG beserta target spesifiknya
* [x] Peran ML dan kontribusi terhadap SDG dijelaskan
* [x] Pertimbangan etika (bias, privasi) dicatat
* [x] Ide, SDG, dan dataset disetujui dosen (**Checkpoint 1**)

---

## MINGGU 2 — [L] DATA UNDERSTANDING & DATA PREPARATION (BAB 4–5)

**Tujuan:** Data bersih, siap dipakai, dan terbagi dengan benar tanpa kebocoran informasi (*data leakage*).

**Tugas:**

1. Isi informasi dataset (jumlah baris/kolom, tipe, ukuran, format) dan tabel deskripsi fitur (Bab 4.1–4.2).
2. Analisis kondisi data: *missing values*, duplikat, *outlier*, *imbalance*, *noise* (Bab 4.3).
3. EDA (opsional): minimal 3 visualisasi beserta *insight* (Bab 4.4).
4. *Data cleaning* dan *feature engineering* (Bab 5.1–5.2).
5. Transformasi data sesuai jenis data (*encoding*, *scaling*, tokenisasi, augmentasi, dll.) (Bab 5.3).
6. **Splitting** train/validation/test dengan `random_state` tetap; gunakan *stratified* jika klasifikasi (Bab 5.4).
7. *Data balancing* jika diperlukan, **hanya pada data training** (Bab 5.5).
8. Tulis ringkasan Apa–Mengapa–Bagaimana (Bab 5.6).
9. Commit notebook `01_learning_problem_eda` dan `02_data_preparation` ke GitHub.

**Deliverable:**
* Laporan Bab 4–5 selesai
* Notebook `01_learning_problem_eda.ipynb` dan `02_data_preparation.ipynb` yang bisa dijalankan ulang di GitHub

**Checklist selesai:**
* [x] Scaler/encoder di-fit hanya pada data training
* [x] Test set disimpan dan tidak disentuh sampai tahap evaluasi akhir
* [x] Jumlah sampel tiap split tercatat
* [x] Setiap langkah preprocessing punya alasan tertulis

---

## MINGGU 3 — [M] MODEL DESIGN: BASELINE & ADVANCED

**Tujuan:** Dua model pertama terlatih dan dijustifikasi berdasarkan hypothesis space dan inductive bias.

**Tugas:**
1. **Model 1 (Baseline):** pilih model sederhana, jelaskan teori, alasan, hypothesis space, dan inductive bias (Bab 6.1.1–6.1.2).
2. Latih baseline, catat hyperparameter dan hasil pada validation set (Bab 6.1.3–6.1.5).
3. **Model 2 (Advanced):** pilih model (Random Forest/XGBoost/SVM/dll.), jelaskan teori, keunggulan, kelemahan, hypothesis space, dan inductive bias (Bab 6.2.1–6.2.2).
4. Latih model advanced dengan parameter awal (Bab 6.2.3).
5. Bandingkan awal baseline vs advanced pada validation set.
6. Commit notebook `03_model_design_baseline_ml`.
7. Commit notebook `03_model_design_advanced_ml`.

**Deliverable:**
- Bab 6.1 dan 6.2.1–6.2.3 selesai
- Skor validasi baseline dan advanced

**Checklist selesai:**
- [ ] Alasan pemilihan model menyebut karakteristik data
- [ ] Asumsi/inductive bias tiap model dijelaskan
- [ ] Hasil validasi tercatat (bukan test set)
- [ ] Asistensi **Checkpoint 2** dilakukan
