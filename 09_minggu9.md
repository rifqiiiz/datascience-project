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

## MINGGU 3 — [M + O + P] MODELING, OPTIMIZATION, EVALUATION & FINALISASI (BAB 6–SELESAI)

**Tujuan:** Merancang, melatih, mengoptimalkan, dan mengevaluasi tiga tingkat model (*Baseline*, *Advanced ML*, dan *Deep Learning* PyTorch) secara objektif serta menyelesaikan laporan akhir proyek beserta artefak repositori.

**Tugas:**

1. **Model Design & Optimization [M + O] (Bab 6):**
   * Implementasikan **Model 1 (Baseline):** *Logistic Regression* (Scikit-Learn) sebagai tolok ukur performa awal.
   * Implementasikan **Model 2 (Advanced ML):** *Random Forest Classifier* (Scikit-Learn) beserta penyetelan *hyperparameter* menggunakan *Validation Set*.
   * Implementasikan **Model 3 (Deep Learning Tabular):** *Multilayer Perceptron* (`MLPModel` subclassing `torch.nn.Module` di **PyTorch**) minimal 2 *hidden layers* dengan aktivasi **ReLU**, `Dropout`, `nn.CrossEntropyLoss()`, *mini-batch* `DataLoader`, dan *optimizer* **Adam** ($\ge 10$ epochs).
   * Catat kurva konvergensi (*training loss* vs *validation loss*) dan eksperimen optimasi (*learning rate*, regularisasi).
2. **Predict & Evaluate [P] (Bab 7):**
   * Evaluasi seluruh model terlatih pada **Test Set** (yang baru dibuka satu kali di tahap akhir) menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score (Macro)*, dan *Confusion Matrix*.
   * Lakukan analisis komparasi antar-model, *error analysis* pada sampel salah prediksi, serta refleksi *ML Lifecycle (L-M-O-P)*.
3. **Finalisasi Laporan & Reproducibility (Bab 8–Selesai):**
   * Lengkapi **Bab 8** (Kesimpulan, Saran, dan Kontribusi terhadap SDG 11.2), **Bab 9** (Daftar Pustaka), serta **Bab 10–12** (Lampiran Struktur Repositori, Panduan Reproduksi `requirements.txt`, dan Bukti Running).
   * Commit notebook pemodelan dan evaluasi (`03_modeling_and_evaluation.ipynb`), simpan bobot model terbaik (`models/`), serta perbarui `README.md` dan laporan lengkap di GitHub.

**Deliverable:**
* Laporan lengkap dari Bab 6 sampai selesai (Bab 6–12)
* Notebook `03_modeling_and_evaluation.ipynb` dengan output pelatihan, kurva loss PyTorch, dan hasil evaluasi akhir
* Repositori GitHub terstruktur dan siap direproduksi (**Checkpoint Final**)

**Checklist selesai:**
* [ ] Ketiga model (Baseline, Advanced ML, Deep Learning PyTorch MLP) berhasil dilatih dan dibandingkan
* [ ] Evaluasi akhir pada Test Set dilakukan hanya sekali tanpa *data leakage*
* [ ] Analisis kesalahan (*error analysis*) dan keterkaitan hasil dengan SDG 11.2 dijelaskan
* [ ] Seluruh kode, dokumentasi laporan lengkap (Bab 1–Selesai), dan `requirements.txt` terunggah rapi di GitHub
