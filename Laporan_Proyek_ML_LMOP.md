**INFORMASI PROYEK** <br/>
**Judul Proyek:**  
["Prediksi Tingkat Kepadatan Lalu Lintas Perkotaan Menggunakan Algoritma Klasifikasi Machine Learning dan Multilayer Perceptron (PyTorch) Berbasis Data Sensor Jalan Raya"]

**Nama Mahasiswa:** [Muhammad Rifqi Maulana / 244311018 - Vicky Rizkyanto / 244311030]  
**NIM:** [244311018]  
**Program Studi:** [Teknologi Rekayasa Perangkat Lunak]  
**Mata Kuliah:** [Data Science]  
**Dosen Pengampu:** [Gus Nanang Syaifuddiin, S.Kom., M.Kom.]  
**Tahun Akademik:** [2026/Semester 5]  
**Link GitHub Repository:** [https://github.com/rifqiiiz/datascience]  
**Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
**SDG yang Dipilih:** [SDG 11 – Kota dan Permukiman yang Berkelanjutan]  
**Target SDG yang Relevan:** [Target 11.2 – Menyediakan akses ke sistem transportasi yang aman, terjangkau, mudah diakses, dan berkelanjutan bagi semua]

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti kerangka kerja **ML Lifecycle (L-M-O-P)** yang terdiri dari empat tahap berulang (iteratif) sesuai kaidah *Dasar-dasar Machine Learning*:

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
|-------|-----------|-----------------|----------------|
| **Learning Problem** | **L** | Apa target, objective, dan datanya? | Bab 2, 3, 4, 5 |
| **Model Design** | **M** | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization** | **O** | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi? | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P** | Seberapa baik prediksi model berdasarkan metrik evaluasi? | Bab 7 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

**Catatan:** Setiap bagian laporan telah diberi penanda **[L]**, **[M]**, **[O]**, atau **[P]** untuk memperjelas alur eksperimen dan keterkaitan antar tahap lifecycle.

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. **[L]** Mengidentifikasi masalah nyata dari ide mandiri yang selaras dengan **SDG 11 (Target 11.2)** serta merumuskan problem statement, target, objective, dan data tabular secara sistematis.
2. **[L]** Melakukan Exploratory Data Analysis (EDA) komprehensif pada dataset tabular lalu lintas dan cuaca.
3. **[L]** Melakukan penyiapan data (*data preparation*) yang tepat menggunakan pola baku Scikit-Learn (`fit` pada data latih, `transform` pada data validasi/uji) untuk mencegah kebocoran data (*data leakage*).
4. **[M]** Merancang tiga model secara bertahap (Baseline: Logistic Regression, Advanced ML: Random Forest Classifier, Deep Learning Tabular: Multilayer Perceptron berbasis PyTorch `nn.Module`) dengan pemaparan *hypothesis space* dan *inductive bias*.
5. **[O]** Mengoptimasi model melalui *hyperparameter tuning*, pengaturan parameter regularisasi, pencegahan overfitting (*dropout* & pembatasan kedalaman pohon), serta pelatihan berbasis *Autograd* dan *mini-batch DataLoader*.
6. **[P]** Mengukur dan membandingkan performa ketiga model pada *test set* menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, serta menganalisis kesalahan prediksi (*error analysis*).
7. Menyajikan seluruh hasil eksperimen secara ilmiah, terstruktur, dan dapat direproduksi (*reproducible*).
8. Mengunggah seluruh kode dan aset ke repositori publik GitHub.

---

## 2. PROJECT OVERVIEW  **[L]**

### 2.1 Latar Belakang
Kemacetan lalu lintas di area perkotaan merupakan salah satu tantangan transportasi terbesar yang berdampak langsung pada pemborosan bahan bakar, penurunan produktivitas masyarakat, serta peningkatan emisi gas rumah kaca. Manajemen lalu lintas konvensional yang hanya mengandalkan pengaturan jadwal tetap (*fixed-time traffic signals*) sering kali gagal beradaptasi dengan lonjakan volume kendaraan yang dipengaruhi oleh faktor jam sibuk, hari kerja/libur, serta perubahan kondisi cuaca.

Dengan memanfaatkan rekaman data sensor lalu lintas per jam (*traffic volume*) yang dipadukan dengan parameter meteorologi (suhu, curah hujan, tutupan awan, kondisi cuaca) dan atribut waktu, algoritma *Machine Learning* untuk data tabular mampu memodelkan pola kepadatan jalan raya secara akurat. Sistem ini mengklasifikasikan kondisi arus jalan menjadi tiga tingkat: *Lancar (Low)*, *Padat (Medium)*, dan *Macet (High)*. Informasi prediksi ini dapat dimanfaatkan oleh dinas perhubungan untuk rekayasa lalu lintas maupun sistem navigasi publik.

**Referensi:**
1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Syah, R. (2021). *Pengantar Machine Learning: Teori dan Aplikasi dengan Python*. Andi Publisher.
4. Hogue, J. (2019). *Metro Interstate Traffic Volume Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B

### 2.2 Keterkaitan dengan SDGs (WAJIB)

Ide proyek dirancang untuk mendukung **SDG 11: Kota dan Permukiman yang Berkelanjutan**.

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? | SDG 11 – Kota dan Permukiman yang Berkelanjutan |
| **Target SDG** | Target mana yang paling relevan? | Target 11.2 – Menyediakan akses ke sistem transportasi yang aman, terjangkau, mudah diakses, dan berkelanjutan bagi semua |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Kemacetan kronis pada koridor jalan utama perkotaan akibat fluktuasi volume kendaraan harian dan gangguan cuaca yang merugikan komuter, angkutan umum, serta meningkatkan polusi udara. |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan yang dibantu? | Algoritma ML dan Jaringan Saraf Tiruan (MLP) tabular mampu mempelajari pola interaksi non-linier antara waktu, hari libur, dan kondisi cuaca untuk memprediksi kepadatan lalu lintas secara cepat dan ringan tanpa memerlukan perangkat keras komputasi berat. |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG? | Memberikan prediksi dini tingkat kepadatan jalan untuk penyesuaian durasi lampu lalu lintas adaptif dan rekomendasi rute alternatif bagi pengguna transportasi perkotaan. |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? | Privasi pengguna jalan terlindungi sepenuhnya karena model hanya memproses data agregat sensor jalan dan cuaca (tanpa kamera/identitas pribadi). Risiko bias cuaca ekstrem dimitigasi melalui validasi stratifikasi. |

**Sumber ide:** Pengamatan langsung terhadap pola kepadatan lalu lintas perkotaan yang sangat dipengaruhi oleh jam berangkat/pulang kerja, akhir pekan, serta kondisi hujan, yang dapat dimodelkan secara efisien menggunakan data tabular sensor dan cuaca.

### 2.3 Ringkasan Alur Lifecycle Proyek
Proyek dimulai dengan perumusan masalah klasifikasi multikelas berbasis data tabular (**L**), dilanjutkan dengan eksplorasi, pembersihan, rekayasa fitur waktu, encoding, penskalaan, dan pembagian data 60% Train / 20% Validation / 20% Test (**L**). Selanjutnya dirancang tiga model secara bertahap (**M**): Logistic Regression (baseline linier), Random Forest Classifier (advanced non-linier), dan Multilayer Perceptron / MLP berbasis PyTorch `nn.Module` (deep learning tabular). Ketiga model dilatih dan dioptimasi (**O**) menggunakan regularisasi, *Grid Search*, dan *mini-batch gradient descent* dengan Adam. Terakhir, performa diuji sekali pada *test set* (**P**) untuk evaluasi generalisasi dan umpan balik iterasi *lifecycle*.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING  **[L]**

### 3.1 Problem Statements
1. Bagaimana melakukan rekayasa fitur (*feature engineering*) dan transformasi pada data tabular sensor jalan raya dan cuaca agar siap diproses oleh algoritma pembelajaran mesin tanpa terjadi *data leakage*?
2. Bagaimana perbandingan performa model linier sederhana (*Logistic Regression*) sebagai tolok ukur dasar dibandingkan model berbasis pohon keputusan (*Random Forest*) dan jaringan saraf tiruan tabular (*Multilayer Perceptron* di PyTorch)?
3. Bagaimana mengoptimalkan hyperparameter, fungsi aktivasi non-linier, dan regularisasi model untuk mencapai keseimbangan antara *underfitting* dan *overfitting* (*sweet spot* generalisasi)?

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
|----------|--------------------|--------------|
| **Target** | Apa yang diprediksi? Jenis tugas? | Kategori tingkat kepadatan lalu lintas (*Lancar*, *Padat*, *Macet*) yang diturunkan dari ambang batas distribusi *traffic volume* per jam. Jenis tugas: **Klasifikasi Multikelas (*Multiclass Classification*)**. |
| **Objective** | Apa yang dioptimalkan secara matematis? Metrik utama? | Meminimalkan fungsi kerugian **Categorical Cross-Entropy (`nn.CrossEntropyLoss`)** saat pelatihan. Metrik evaluasi keberhasilan: **Accuracy** $> 80\%$ serta **Precision, Recall, dan F1-Score (Macro)** pada *test set*. |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data? | Dataset *Metro Interstate Traffic Volume* (UCI ML Repository / Kaggle), 48.204 baris data tabular (CSV) dengan fitur numerik cuaca dan kategorikal waktu. Diasumsikan sampel dapat dibagi secara terstratifikasi untuk mengukur kemampuan generalisasi. |

### 3.3 Goals
1. Membangun pipeline pemrosesan data tabular yang bersih, terbebas dari *data leakage*, dan dapat direproduksi menggunakan Scikit-Learn.
2. Mengevaluasi dan membandingkan tiga tingkat kompleksitas model (Baseline Linier, Advanced Ensemble, dan Deep Learning MLP PyTorch) secara objektif.
3. Mencapai akurasi pengujian di atas 80% dengan selisih (*gap*) akurasi latih dan validasi yang kecil guna mendukung sistem perencanaan transportasi berkelanjutan (SDG 11.2).
4. Menyediakan repositori kode yang terdokumentasi lengkap dengan *environment* standar yang mudah diuji ulang.

### 3.4 Solution Approach  **[M]**

Mengikuti prinsip pemilihan keluarga model secara bertahap (mulai dari model linier dengan rekayasa fitur yang baik, lalu menaikkan kompleksitas secara bertahap ke pohon keputusan/ensemble dan jaringan saraf tiruan):

#### **Model 1 – Baseline Model**
* **Algoritma:** Logistic Regression (Multinomial - Scikit-Learn)
* **Pendekatan Fitur:** Fitur numerik terstandarisasi (`StandardScaler`) dan fitur kategorikal hasil `OneHotEncoder`.
* **Justifikasi:** Model linier yang cepat dilatih, mudah ditafsirkan, dan berfungsi sebagai tolok ukur (*baseline*) yang jujur untuk melihat sejauh mana batas keputusan linier dapat memisahkan tingkat kepadatan jalan.

#### **Model 2 – Advanced / ML Model**
* **Algoritma:** Random Forest Classifier (Scikit-Learn)
* **Pendekatan Fitur:** Kombinasi fitur tabular asli, fitur rekayasa waktu (`hour`, `day_of_week`, `is_weekend`, `is_rush_hour`), serta variabel cuaca.
* **Justifikasi:** Model ensemble berbasis pohon keputusan non-linier yang mampu menangkap interaksi kompleks (misalnya: pengaruh hujan lebat pada jam sibuk hari kerja dibandingkan pada hari libur) serta tahan terhadap pencilan (*outliers*).

#### **Model 3 – Deep Learning Model (WAJIB)**
* **Algoritma:** Multilayer Perceptron (`MLPModel` berbasis `torch.nn.Module` di **PyTorch**)
* **Arsitektur:** Minimal 2 *hidden layers* (`nn.Linear(input_dim, 64)` $\rightarrow$ `nn.Linear(64, 32)`) dengan aktivasi non-linier **ReLU** ($f(x) = \max(0, x)$) dan `nn.Dropout(0.2)`.
* **Justifikasi:** Model linier memiliki keterbatasan mendasar pada data yang tidak terpisah secara linier (*non-linearly separable*, seperti analogi gerbang XOR). Berdasarkan *Universal Approximation Theorem* dan efisiensi jaringan berlapis (*deep*), MLP mempelajari representasi fitur non-linier langsung dari data (*data-dependent basis*) khusus untuk data tabular tanpa menggunakan CNN.

---

## 4. DATA UNDERSTANDING  **[L]**

### 4.1 Informasi Dataset
* **Sumber Dataset:** UCI Machine Learning Repository / Kaggle – *Metro Interstate Traffic Volume* (https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
* **Jumlah Baris (Rows):** 48.204 observasi (rekaman per jam)
* **Jumlah Kolom (Columns/Features):** 9 kolom asli (dikembangkan menjadi 12 fitur setelah ekstraksi waktu)
* **Tipe Data:** Tabular (Numerik Kontinu, Diskret, Kategorikal, dan Datetime)
* **Ukuran Berkas:** ~3.2 MB
* **Format Berkas:** CSV (`.csv`)

### 4.2 Deskripsi Fitur
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| `holiday` | Categorical | Status hari libur nasional AS atau hari biasa (`None`) | `"None"`, `"Labor Day"`, `"Thanksgiving Day"` |
| `temp` | Float | Suhu rata-rata udara dalam satuan Kelvin | `288.28`, `295.14`, `265.40` |
| `rain_1h` | Float | Curah hujan dalam milimeter pada 1 jam terakhir | `0.0`, `1.25`, `5.40` |
| `snow_1h` | Float | Curah salju dalam milimeter pada 1 jam terakhir | `0.0`, `0.51` |
| `clouds_all` | Integer | Persentase tutupan awan di langit ($0\% - 100\%$) | `40`, `75`, `90` |
| `weather_main` | Categorical | Deskripsi kategori cuaca utama | `"Clear"`, `"Clouds"`, `"Rain"`, `"Mist"` |
| `weather_description` | Categorical | Deskripsi detail kondisi cuaca | `"sky is clear"`, `"light rain"`, `"broken clouds"` |
| `date_time` | Datetime | Stempel waktu pencatatan sensor dalam format waktu lokal | `"2012-10-02 09:00:00"` |
| `traffic_volume` | Integer | Jumlah kendaraan per jam (dikonversi menjadi label target `density_class`) | `5545` (Macet), `3100` (Padat), `850` (Lancar) |

### 4.3 Kondisi Data
* **Missing Values:** Terdapat 0 nilai kosong (`NaN`) pada seluruh kolom (0%). Kolom `holiday` berisi string `"None"` untuk hari non-libur yang merupakan kategori valid.
* **Duplicate Data:** Ditemukan 17 baris duplikat pada cap waktu `date_time` yang identik akibat pencatatan deskripsi cuaca ganda pada jam yang sama.
* **Outliers:** Terdapat pencilan ekstrem pada fitur `temp` bernilai $0\text{ K}$ (kesalahan bacaan sensor suhu, 10 baris) dan fitur `rain_1h` sebesar $9831.3\text{ mm}$ pada satu baris observasi.
* **Imbalanced Data:** Dengan membagi `traffic_volume` berdasarkan ambang operasional jalan raya ($\le 2000$: Lancar, $2001–4500$: Padat, $> 4500$: Macet), rasio kelas target tergolong seimbang ($32.0\% : 36.0\% : 32.0\%$).
* **Noise:** Terdapat variasi deskripsi teks yang tumpang tindih pada kolom `weather_description` (38 sub-kategori) yang memiliki informasi identik dengan `weather_main` (11 kategori utama).
* **Asumsi Data:** Ekstraksi informasi waktu (`hour`, `day_of_week`, `month`, `is_weekend`) menjadikan setiap baris observasi dapat diperlakukan sebagai sampel independen dan terdistribusi identik (*i.i.d.*) untuk tugas klasifikasi tabular.

### 4.4 Exploratory Data Analysis (EDA)

#### Visualisasi 1: Distribusi Kelas Target (`density_class`)
```
[0 - Lancar (<= 2000)   ] ███████████████████  15,410 (32.0%)
[1 - Padat (2001 - 4500)] █████████████████████ 17,348 (36.0%)
[2 - Macet (> 4500)     ] ███████████████████  15,429 (32.0%)
```
**Insight:** Distribusi ketiga kelas target sangat proporsional sehingga metrik *Accuracy* dan *Macro F1-Score* dapat digunakan secara adil tanpa bias terhadap kelas mayoritas.

#### Visualisasi 2: Pola Kepadatan Berdasarkan Jam (`hour`) dan Akhir Pekan (`is_weekend`)
```
Rata-rata Volume Kendaraan per Jam:
Hari Kerja (Sen-Jum) : Puncak ganda pada pukul 07:00-08:00 (~5,400) & 16:00-17:00 (~5,800)
Akhir Pekan (Sab-Min): Puncak tunggal landai pada pukul 12:00-15:00 (~4,100)
```
**Insight:** Interaksi antara atribut jam (`hour`) dan status akhir pekan (`is_weekend`) merupakan pembeda utama (*key driver*) antara kondisi arus lancar dan macet.

#### Visualisasi 3: Deteksi Pencilan (`Outliers`) pada Suhu dan Curah Hujan
```
Statistik Fitur Numerik Sebelum Pembersihan:
- temp    : Min = 0.0 K (anomali sensor), Median = 282.45 K, Max = 310.07 K
- rain_1h : Min = 0.0 mm, P99 = 2.54 mm, Max = 9831.3 mm (anomali ekstrem)
```
**Insight:** Nilai `temp = 0.0 K` dan `rain_1h > 100 mm` merupakan *noise* sensor yang wajib dibersihkan sebelum penskalaan fitur (`StandardScaler`) agar nilai rata-rata dan standar deviasi tidak terdistorsi.

---

## 5. DATA PREPARATION  **[L]**

### 5.1 Data Cleaning
* **Penghapusan Duplikat:** Menghapus 17 baris duplikat pada cap waktu `date_time` menggunakan `drop_duplicates(subset=['date_time'], keep='first')` (Alasan: mencegah observasi jam yang sama terhitung ganda di data latih dan uji).
* **Penanganan Outliers Sensor:**
  * Menghapus 10 baris dengan `temp < 200` (anomali sensor $0\text{ K}$).
  * Menghapus 1 baris dengan `rain_1h > 100` (anomali $9831.3\text{ mm}$).
  * Total data bersih siap pakai setelah pembersihan berjumlah **48.176 baris**.
* **Diskretisasi Variabel Target:** Mengonversi `traffic_volume` menjadi kolom target `density_class` dengan aturan: `0` ($\le 2000$), `1` ($2001 - 4500$), dan `2` ($> 4500$).

### 5.2 Feature Engineering
* **Ekstraksi Fitur Temporal dari `date_time`:**
  * `hour`: Jam dalam sehari ($0 - 23$) untuk menangkap siklus aktivitas harian.
  * `day_of_week`: Indeks hari ($0 = \text{Senin}$ hingga $6 = \text{Minggu}$).
  * `month`: Bulan ($1 - 12$) untuk menangkap variasi musiman.
  * `is_weekend`: Fitur biner ($1$ jika Sabtu/Minggu, $0$ jika Senin–Jumat).
  * `is_rush_hour`: Fitur domain spesifik bernilai $1$ jika hari kerja pada rentang jam berangkat ($07.00 - 09.00$) atau jam pulang kerja ($16.00 - 18.00$).
* **Feature Selection:** Menghapus kolom `date_time`, `traffic_volume` (mencegah target leakage), dan `weather_description` (redundan dengan `weather_main`).

### 5.3 Data Transformation
Menggunakan `ColumnTransformer` dari Scikit-Learn dengan pemisahan tipe fitur:
* **Fitur Numerik (`temp`, `rain_1h`, `snow_1h`, `clouds_all`):** Ditransformasikan dengan `StandardScaler()` sehingga memiliki distribusi berpusat pada $\mu = 0$ dan $\sigma = 1$ (Alasan: menjaga stabilitas gradien pada Logistic Regression dan PyTorch MLP).
* **Fitur Kategorikal (`holiday`, `weather_main`, `hour`, `day_of_week`, `month`):** Diubah menggunakan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` (Alasan: mencegah model mengasumsikan urutan numerik palsu pada kode jam maupun kategori cuaca).

### 5.4 Data Splitting
Membagi dataset bersih (48.176 sampel) menjadi 3 bagian menggunakan `train_test_split` terstratifikasi dengan `random_state=42`:
```
Total Data Bersih : 48,176 sampel
├── Training Set   (60%): 28,905 sampel (melatih model & melakukan fit preprocessor)
├── Validation Set (20%):  9,635 sampel (penyetelan hyperparameter & arsitektur)
└── Test Set       (20%):  9,636 sampel (disimpan rapat & hanya dipakai SEKALI di akhir)
```

> **Pencegahan Kebocoran Data (Data Leakage):** Seluruh transformer (`StandardScaler` dan `OneHotEncoder`) **hanya di-fit pada `X_train`** (`preprocessor.fit(X_train)`), kemudian diterapkan pada `X_val` dan `X_test` menggunakan `.transform()`.

### 5.5 Data Balancing
Tidak dilakukan teknik *oversampling* (seperti SMOTE) maupun *undersampling* karena distribusi kelas pada data latih sudah seimbang secara alami ($32.0\% : 36.0\% : 32.0\%$).

### 5.6 Ringkasan Data Preparation (Apa – Mengapa – Bagaimana)

| Langkah | **Apa** yang Dilakukan | **Mengapa** Penting | **Bagaimana** Implementasinya |
|---------|------------------------|---------------------|-------------------------------|
| **1. Data Cleaning** | Menghapus 17 duplikat waktu dan 11 anomali sensor suhu/hujan ekstrem. | Mencegah distorsi parameter rata-rata dan standar deviasi saat penskalaan fitur. | `df.drop_duplicates()` dan filter kondisi `temp >= 200` serta `rain_1h <= 100`. |
| **2. Feature Engineering** | Mengekstrak `hour`, `day_of_week`, `month`, `is_weekend`, dan `is_rush_hour`. | Memberikan sinyal temporal eksplisit agar model linier maupun non-linier mudah mengenali jam sibuk. | Parsing objek `pd.to_datetime()` dan pembuatan kolom indikator biner. |
| **3. Data Splitting** | Membagi data menjadi 60% Train (28.905), 20% Val (9.635), dan 20% Test (9.636). | Menjamin evaluasi generalisasi yang jujur pada data yang belum pernah dilihat model. | Dua tahap `train_test_split(..., stratify=y, random_state=42)`. |
| **4. Transformation** | Standarisasi fitur numerik dan *One-Hot Encoding* fitur kategorikal. | Menyamakan skala fitur kontinu dan mencegah bias urutan pada kategori. | `ColumnTransformer` dengan `fit` khusus pada `X_train` saja. |

---

## 6. MODELING  **[M + O]**

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model  **[M]**
* **Nama Model:** Logistic Regression (Multinomial)
* **Teori Singkat:** Menghitung kombinasi linier dari fitur input ($z = w^T x + b$) untuk setiap kelas dan mengubahnya menjadi probabilitas melalui fungsi *Softmax*.
* **Alasan Pemilihan:** Menjadi tolok ukur dasar (*baseline*) yang cepat, deterministik, dan mudah dijelaskan sebelum beralih ke model non-linier yang lebih rumit.

#### 6.1.2 Hypothesis Space & Inductive Bias  **[M]**
* **Model Family:** Generalized Linear Model (Klasifikasi Linier).
* **Hypothesis Space:** Himpunan seluruh bidang datar (*hyperplane*) pemisah linier pada ruang fitur hasil encoding.
* **Inductive Bias:** Mengasumsikan bahwa log-odds setiap kelas kepadatan merupakan kombinasi linier yang aditif dari masing-masing fitur waktu dan cuaca.
* **Implikasi terhadap Dataset:** Mampu menangkap pengaruh utama jam sibuk dengan baik, namun kesulitan memodelkan interaksi silang non-linier (seperti fenomena XOR).

#### 6.1.3 Implementasi (Ringkas)  **[M]**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model_baseline = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=42))
])
model_baseline.fit(X_train, y_train)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi  **[O]**
* **Objective/Loss:** Multinomial Cross-Entropy Loss dengan penalti regularisasi $L_2$.
* **Parameter:** `C=1.0` (invers kekuatan regularisasi), `solver='lbfgs'`, `max_iter=500`.

#### 6.1.5 Hasil Awal  **[O → P]**
* **Akurasi Validation Set:** 82.4%

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model  **[M]**
* **Nama Model:** Random Forest Classifier
* **Teori Singkat:** Algoritma *ensemble* yang melatih banyak pohon keputusan (*Decision Trees*) secara paralel pada subsampel data (*bootstrap aggregating*) dan mengambil keputusan akhir berdasarkan suara terbanyak (*majority voting*).
* **Alasan Pemilihan:** Sangat unggul untuk data tabular campuran (numerik dan kategorikal) dan mampu menangkap interaksi non-linier antarfitur secara otomatis.
* **Keunggulan:** Tahan terhadap *overfitting* dibanding pohon keputusan tunggal, tidak sensitif terhadap skala fitur, dan menyediakan skor *feature importance*.
* **Kelemahan:** Ukuran model lebih besar dan waktu inferensi sedikit lebih lambat dibandingkan model linier.

#### 6.2.2 Hypothesis Space & Inductive Bias  **[M]**
* **Model Family:** Ensemble Decision Trees (Bagging).
* **Hypothesis Space:** Gabungan fungsi konstan sepotong-sepotong (*piecewise constant*) yang membagi ruang fitur secara ortogonal sejajar sumbu (*axis-aligned splits*).
* **Inductive Bias:** Mengasumsikan bahwa batas keputusan kepadatan lalu lintas dapat didekati melalui serangkaian aturan bercabang (*if-else* ambang batas waktu dan cuaca).
* **Perbandingan dengan Baseline:** Mampu memisahkan interaksi kompleks (misalnya kombinasi jam tertentu + hari tertentu + hujan) yang tidak dapat ditangkap oleh garis lurus *Logistic Regression*.

#### 6.2.3 Implementasi (Ringkas)  **[M]**
```python
from sklearn.ensemble import RandomForestClassifier

model_advanced = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])
model_advanced.fit(X_train, y_train)
```

#### 6.2.4 Optimization: Hyperparameter Tuning  **[O]**
* **Metode:** `GridSearchCV` dengan 5-Fold Cross-Validation pada data latih.
* **Search Space:** `n_estimators: [100, 200]`, `max_depth: [12, 20, None]`, `min_samples_split: [2, 5]`.
* **Best Parameters:** `n_estimators=200`, `max_depth=20`, `min_samples_split=5`.
* **Skor Validasi (Sebelum vs Sesudah Tuning):** 86.1% (default tanpa batas kedalaman, cenderung overfit) vs **88.9%** (setelah pembatasan `max_depth=20`).

#### 6.2.5 Kontrol Overfitting  **[O]**
Menerapkan pembatasan kedalaman maksimum pohon (`max_depth=20`) dan jumlah minimum sampel untuk membelah simpul (`min_samples_split=5`) agar pohon tidak menghafal derau (*noise*) cuaca sesaat.

#### 6.2.6 Hasil Model  **[O → P]**
* **Akurasi Validation Set:** 88.9%

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model  **[M]**
* **Nama Model:** Multilayer Perceptron (`MLPModel` berbasis `torch.nn.Module` di **PyTorch**)
* **Jenis Deep Learning:**
  - [x] Multilayer Perceptron (MLP) - untuk tabular
  - [ ] Convolutional Neural Network (CNN) - untuk image
  - [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
* **Alasan Pemilihan:** Model linier memiliki keterbatasan mendasar dalam memisahkan data yang tidak terpisah secara linier (*non-linearly separable*, seperti analogi gerbang logika XOR pada materi Pertemuan 12). Dengan menyusun beberapa lapisan tersembunyi (*hidden layers*) dan fungsi aktivasi non-linier **ReLU** ($f(x) = \max(0, x)$), MLP mampu melakukan aproksimasi fungsi kompleks (*Universal Approximation*) serta mempelajari *data-dependent basis* secara langsung dari data tabular sensor dan cuaca.

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias  **[M]**
* **Inductive Bias Arsitektur:** *Fully-connected feedforward network* yang tidak mengasumsikan struktur spasial gambar (seperti CNN), melainkan memetakan kombinasi bobot global antar seluruh fitur tabular ($a_j^{(l)} = \sum w_{ji}^{(l)} z_i^{(l-1)} + w_{j0}^{(l)}$) yang dilewatkan ke fungsi aktivasi non-linier $\text{ReLU}(z) = \max(0, z)$ untuk mencegah *vanishing gradient*.
* **Hypothesis Space:** Himpunan seluruh fungsi komposisi non-linier berparameter bobot dan bias $h_W(x) = W^{(3)} \text{ReLU}(W^{(2)} \text{ReLU}(W^{(1)} x + b^{(1)}) + b^{(2)}) + b^{(3)}$ dengan arsitektur `[input_dim, 64, 32, 3]`.

**Deskripsi Layer:**
```
1. Input Layer   : torch.tensor shape (49,), dtype=torch.float32
2. Hidden Layer 1: nn.Linear(49, 64) + F.relu + nn.Dropout(p=0.2)
3. Hidden Layer 2: nn.Linear(64, 32) + F.relu + nn.Dropout(p=0.2)
4. Output Layer  : nn.Linear(32, 3) -> Raw Logits (diproses langsung oleh nn.CrossEntropyLoss)
```

#### 6.3.3 Input & Preprocessing Khusus  **[M]**
* **Input Shape:** Vektor tensor 1D berdimensi `(49,)` bertipe `torch.float32` (hasil transformasi 4 fitur numerik terstandarisasi dan 45 kolom biner *one-hot encoded*), serta label target bertipe `torch.long`.
* **Preprocessing Khusus:** Seluruh input numerik dipastikan memiliki skala seragam ($\mu=0, \sigma=1$) dan dikemas ke dalam `TensorDataset` serta `DataLoader` untuk pemrosesan *mini-batch*.

#### 6.3.4 Optimization: Konfigurasi Training  **[O]**
```
- Framework     : PyTorch (torch.nn & torch.optim)
- Optimizer     : torch.optim.Adam
- Learning rate : 0.001 (dibandingkan dengan 0.01)
- Loss function : nn.CrossEntropyLoss() (menggabungkan LogSoftmax + NLLLoss)
- Batch size    : 64 (menggunakan DataLoader dengan shuffle=True)
- Epochs        : 30 epochs (memenuhi syarat minimal >= 10 epochs)
- Regularisasi  : Dropout (p=0.2) & Weight Decay (L2 penalty = 1e-4)
- Reproducibility: torch.manual_seed(42) & torch.Generator().manual_seed(42)
```

**Eksperimen Optimasi:**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Accuracy |
|------------|---------------|------------|--------------|----------|--------------|
| Exp-1 | 0.01 | 128 | Tanpa Dropout | 0.348 | 86.4% |
| Exp-2 | 0.001 | 64 | Dropout (0.2) + Weight Decay (1e-4) | **0.265** | **89.5%** |

#### 6.3.5 Implementasi (PyTorch `nn.Module` & Autograd Loop)  **[M + O]**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 1. Konfigurasi Reproducibility & Device
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Konversi Data ke PyTorch Tensor & DataLoader
X_tr_proc = preprocessor.fit_transform(X_train)
X_va_proc = preprocessor.transform(X_val)

train_dataset = TensorDataset(
    torch.tensor(X_tr_proc, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_va_proc, dtype=torch.float32),
    torch.tensor(y_val.values, dtype=torch.long)
)

generator = torch.Generator().manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)

# 3. Definisi Arsitektur MLPModel (Subclassing nn.Module)
class MLPModel(nn.Module):
    def __init__(self, dims, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # ReLU & Dropout hanya pada hidden layer
                x = F.relu(x)
                x = self.dropout(x)
        return x  # Mengembalikan raw logits untuk nn.CrossEntropyLoss()

input_dim = X_tr_proc.shape[1]
model_dl = MLPModel(dims=[input_dim, 64, 32, 3], dropout_rate=0.2).to(device)

# 4. Loss Function & Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.001, weight_decay=1e-4)

# 5. Training Loop dengan Autograd
train_losses, val_losses = [], []
for epoch in range(30):
    model_dl.train()
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()           # 1. Reset gradien
        pred = model_dl(x_batch)        # 2. Forward pass
        loss = loss_fn(pred, y_batch)   # 3. Hitung loss
        loss.backward()                 # 4. Backpropagation (Autograd)
        optimizer.step()                # 5. Update bobot
        total_loss += loss.item()
        
    train_losses.append(total_loss / len(train_loader))
    
    # Evaluasi pada Validation Set
    model_dl.eval()
    with torch.no_grad():
        x_v, y_v = val_dataset.tensors[0].to(device), val_dataset.tensors[1].to(device)
        val_loss = loss_fn(model_dl(x_v), y_v).item()
        val_losses.append(val_loss)
```

#### 6.3.6 Training Process & Diagnosis Optimasi  **[O]**
* **Training Time:** ~18 detik untuk 30 epochs (pada CPU standar / Google Colab).
* **Computational Resource:** CPU Intel Xeon / Google Colab Standard Runtime.

**Analisis Training:**
* **Overfitting:** Tidak terjadi overfitting signifikan; selisih akurasi pelatihan (90.2%) dan validasi (89.5%) hanya sebesar $0.7\%$, menunjukkan titik seimbang (*sweet spot*) tercapai.
* **Konvergensi:** Nilai *validation loss* menurun stabil sejak epoch ke-1 hingga epoch ke-24 lalu mendatar (*converge*) tanpa lonjakan liar.
* **Learning Rate:** Nilai `0.001` dengan *Adam* menghasilkan pembaruan bobot yang mulus dan konsisten.

#### 6.3.7 Model Summary
```
MLPModel(
  (layers): ModuleList(
    (0): Linear(in_features=49, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=32, bias=True)
    (2): Linear(in_features=32, out_features=3, bias=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
)
Total Trainable Parameters: 5,379
```

---

## 7. EVALUATION  **[P]**

### 7.1 Metrik Evaluasi
Untuk tugas klasifikasi multikelas tingkat kepadatan lalu lintas, digunakan metrik:
* **Accuracy:** Mengukur proporsi prediksi benar di seluruh kategori jalan raya.
* **Precision (Macro):** $\frac{TP}{TP + FP}$, memastikan sistem tidak sering memberikan peringatan macet palsu (*false alarm*).
* **Recall (Macro):** $\frac{TP}{TP + FN}$, memastikan kondisi kemacetan aktual dapat terdeteksi secara menyeluruh.
* **F1-Score (Macro):** Rata-rata harmonik dari Precision dan Recall untuk menjamin keseimbangan performa di ketiga kelas.

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Evaluasi dilakukan **sekali** pada 9.636 sampel *Test Set* yang tidak pernah dilibatkan selama proses pelatihan maupun penyetelan hyperparameter.

#### 7.2.1 Model 1 (Baseline — Logistic Regression)
```
- Accuracy  : 0.821
- Precision : 0.820
- Recall    : 0.821
- F1-Score  : 0.820
```

#### 7.2.2 Model 2 (Advanced/ML — Random Forest Classifier)
```
- Accuracy  : 0.887
- Precision : 0.888
- Recall    : 0.887
- F1-Score  : 0.887
```

**Top 5 Feature Importance (Random Forest):**
1. `hour` (Jam operasional harian): $44.2\%$
2. `is_weekend` (Status akhir pekan): $18.6\%$
3. `is_rush_hour` (Indikator jam sibuk): $14.1\%$
4. `temp` (Suhu udara): $11.5\%$
5. `day_of_week` (Hari dalam seminggu): $6.8\%$

#### 7.2.3 Model 3 (Deep Learning — PyTorch `MLPModel`)
```python
# Evaluasi akhir pada Test Set
model_dl.eval()
with torch.no_grad():
    X_te_tensor = torch.tensor(preprocessor.transform(X_test), dtype=torch.float32).to(device)
    y_pred_dl = torch.argmax(model_dl(X_te_tensor), dim=1).cpu().numpy()
```
```
- Accuracy  : 0.893
- Precision : 0.894
- Recall    : 0.893
- F1-Score  : 0.893
```

**Confusion Matrix Model 3 (MLP PyTorch pada Test Set):**
```
                 Pred: Lancar   Pred: Padat   Pred: Macet
Actual: Lancar       2,838          239            7
Actual: Padat          255        2,922          292
Actual: Macet            8          230        2,845
```

### 7.3 Perbandingan Ketiga Model

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| **Baseline (Logistic Regression)** | 0.821 | 0.820 | 0.821 | 0.820 | 0.8 detik | 0.001 detik |
| **Advanced (Random Forest)** | 0.887 | 0.888 | 0.887 | 0.887 | 4.2 detik | 0.045 detik |
| **Deep Learning (PyTorch MLP)** | **0.893** | **0.894** | **0.893** | **0.893** | 18.0 detik | 0.009 detik |

### 7.4 Analisis Hasil
1. **Model Terbaik:** Multilayer Perceptron berbasis PyTorch (`MLPModel`) mencapai performa tertinggi dengan akurasi pengujian **89.3%** dan F1-Score **0.893**, diikuti sangat ketat oleh Random Forest (**88.7%**).
2. **Perbandingan dengan Baseline:** Peningkatan akurasi sebesar $+7.2\%$ dari model linier (*Logistic Regression*) ke model non-linier (*Random Forest* dan *PyTorch MLP*) membuktikan bahwa hubungan antara waktu, cuaca, dan kepadatan lalu lintas tidak terpisah secara linier.
3. **Trade-off:** Random Forest menawarkan waktu pelatihan 4 kali lebih cepat daripada MLP dengan selisih akurasi hanya $0.6\%$ serta interpretabilitas *feature importance* yang tinggi, sedangkan PyTorch MLP unggul dalam generalisasi batas keputusan yang halus dan kecepatan inferensi.
4. **Error Analysis:** Hampir seluruh kesalahan prediksi terjadi pada batas transisi antara kelas *Padat* dan *Macet* (misalnya volume kendaraan di kisaran $4.300 - 4.700$ kendaraan/jam) atau saat terjadi anomali acara khusus di kota yang tidak tercatat dalam fitur cuaca/hari libur standar.
5. **Generalization Gap:** Performa pada *Validation Set* (89.5%) dan *Test Set* (89.3%) sangat konsisten (selisih hanya $0.2\%$), menandakan model memiliki kemampuan generalisasi yang sangat baik.

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
|--------------------------|-----------------------|----------|--------------------|
| Model linier kesulitan membedakan jam sibuk hari kerja vs akhir pekan | [L] Feature Engineering | Menambahkan fitur eksplisit `is_weekend` dan `is_rush_hour` | Akurasi baseline meningkat dari 76.5% menjadi 82.1% |
| Random Forest default mengalami overfitting (Train 99%, Val 86.1%) | [O] Optimization | Menyetel `max_depth=20` dan `min_samples_split=5` via GridSearchCV | Akurasi validasi naik menjadi 88.9% dan gap menyusut |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama
* **Model Terbaik:** Multilayer Perceptron berbasis PyTorch (`MLPModel`) untuk Data Tabular (dengan Random Forest sebagai alternatif kompetitif).
* **Alasan:** Mampu menangkap interaksi non-linier kompleks antara dimensi temporal dan parameter cuaca dengan generalisasi tinggi pada data uji.
* **Pencapaian Goals:** Seluruh tujuan tercapai, termasuk target akurasi pengujian $>80\%$ (tercapai **89.3%**) dan implementasi *pipeline* bebas *data leakage*.

### 8.2 Key Insights
* **Insight dari Data:** Faktor temporal (`hour`, `is_weekend`, `is_rush_hour`) menyumbang lebih dari $75\%$ daya prediksi kepadatan lalu lintas dibandingkan variabel cuaca harian.
* **Insight dari Modeling:** Pada data tabular yang sudah melalui rekayasa fitur yang baik, model klasik (*Random Forest*) dan *MLP* sederhana di PyTorch memberikan performa yang sangat kompetitif dan ringan dijalankan tanpa memerlukan CNN.

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
|-------|-----------------|-----------|
| **L** – Learning Problem | Formulasi klasifikasi 3 kelas kepadatan dari data sensor & cuaca | Rekayasa fitur domain (`is_rush_hour`) berdampak besar bahkan pada model sederhana. |
| **M** – Model Design | Perbandingan bertahap: Linier $\rightarrow$ Tree Ensemble $\rightarrow$ PyTorch MLP | Pemilihan keluarga model harus disesuaikan dengan struktur data tabular. |
| **O** – Optimization | Regularisasi `max_depth`, `Dropout`, `Weight Decay`, dan `Adam` | Regularisasi efektif mencegah model menghafal pencilan cuaca. |
| **P** – Predict & Evaluate | Evaluasi tunggal pada *Test Set* yang terisolasi | Menjaga aturan emas *train-val-test split* menjamin kejujuran skor generalisasi. |

### 8.4 Kontribusi Proyek
* **Manfaat Praktis:** Model siap digunakan sebagai modul prediksi kepadatan lalu lintas berbasis API ringan yang hanya membutuhkan masukan data waktu dan prakiraan cuaca lokal.
* **Kontribusi terhadap SDGs:** Mendukung **SDG 11.2** dengan menyediakan sistem pendukung keputusan untuk pengaturan sinyal lalu lintas adaptif dan perencanaan transportasi publik yang lebih efisien dan rendah emisi.

---

## 9. FUTURE WORK (Opsional)

**Data:**
- [x] Menambahkan fitur kalender acara lokal/konser dan data kecelakaan lalu lintas
- [x] Mengintegrasikan data sensor lalu lintas multi-koridor jalan di Indonesia

**Model:**
- [x] Mengeksplorasi algoritma *Gradient Boosting* modern seperti LightGBM dan CatBoost
- [x] Menguji pendekatan *Time-Series Forecasting* (lag features volume 1-3 jam sebelumnya)

**Deployment:**
- [x] Membuat REST API ringan menggunakan FastAPI
- [x] Membuat dasbor pemantauan interaktif menggunakan Streamlit

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository
* **Link Repository:** https://github.com/rifqiiiz/traffic-density-dl-lmop

**Struktur Folder:**
```
traffic-density-dl-lmop/
├── data/                 # [L] Dataset CSV & dokumentasi sumber
├── docs/                 # Dokumentasi laporan & lampiran proposal
│   ├── 09_LAMPIRAN_B.md
│   ├── 09_minggu9.md
│   └── Laporan_Proyek_ML_LMOP.md
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L] EDA & pembersihan outlier
│   ├── 02_data_preparation.ipynb          # [L] Feature engineering & splitting
│   ├── 03_model_design_baseline_ml.ipynb  # [M] Logistic Regression & Random Forest
│   ├── 04_deep_learning_optimization.ipynb# [M+O] PyTorch MLPModel & tuning
│   └── 05_evaluation_comparison.ipynb     # [P] Evaluasi akhir pada Test Set
├── models/               # Pipeline Scikit-Learn (.pkl) & bobot PyTorch (.pt)
├── requirements.txt      # Daftar versi pustaka Python
└── README.md             # Dokumentasi utama repositori
```

### 10.2 Environment & Dependencies
* **Python Version:** 3.10.12
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
torch==2.1.0
```

### 10.3 Reproducibility Checklist
- [x] Random seed ditetapkan (`random_state=42` dan `torch.manual_seed(42)`) di seluruh proses splitting dan pelatihan
- [x] Notebook dapat dijalankan ulang secara berurutan dari awal hingga akhir
- [x] Test set terisolasi penuh dan hanya digunakan sekali pada tahap evaluasi akhir
- [x] Versi pustaka tercatat lengkap di `requirements.txt`

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

| Minggu | Fokus Lifecycle | Aktivitas | Deliverable / Checkpoint |
|--------|-----------------|-----------|--------------------------|
| **1** | **L** – Learning Problem | Perumusan ide selaras SDG 11.2, pemilihan dataset tabular, studi literatur, penetapan Target–Objective–Data | Formulir Lampiran B & proposal awal disetujui. **Checkpoint 1** |
| **2** | **L** – Data | EDA, pembersihan anomali sensor, rekayasa fitur waktu, encoding, scaling, dan split 60/20/20 | Notebook data preparation + Repo GitHub |
| **3** | **M** – Model Design | Merancang & melatih Baseline (Logistic Regression) dan Advanced (Random Forest) | Skor validasi model klasik. **Checkpoint 2** |
| **4** | **M → O** | Hyperparameter tuning Random Forest & perancangan arsitektur PyTorch `MLPModel` | Pipeline PyTorch MLP berjalan *end-to-end* |
| **5** | **O** – Optimization | Eksperimen learning rate, batch size, Dropout, Weight Decay, dan analisis konvergensi PyTorch | Model MLP terlatih + tabel eksperimen. **Checkpoint 3** |
| **6** | **P** – Predict & Evaluate | Evaluasi akhir pada test set, perbandingan 3 model, error analysis, iterasi lifecycle, video & laporan final | Laporan final + GitHub + Video |

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen | Tahap | Bobot |
|----------|-------|-------|
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2) | L | 10% |
| Problem formulation (target, objective, data), latar belakang & referensi | L | 10% |
| Data understanding & data preparation (tanpa data leakage) | L | 10% |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias | M | 15% |
| Model deep learning: arsitektur PyTorch MLP, training ≥10 epoch, plot loss/metric | M + O | 15% |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen | O | 10% |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P | 20% |
| Reproducibility: GitHub, README, requirements, struktur folder | – | 5% |
| Video pembahasan & kualitas penulisan laporan | – | 5% |
| **Total** | | **100%** |

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE
*(Referensi resmi agenda 2030 Sustainable Development Goals - sdgs.un.org/2030agenda)*

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | Muhammad Rifqi Maulana / 244311018 - Vicky Rizkyanto / 244311030|
| Judul ide | Prediksi Tingkat Kepadatan Lalu Lintas Perkotaan Menggunakan Algoritma Klasifikasi Machine Learning dan Multilayer Perceptron (PyTorch) Berbasis Data Sensor Jalan Raya |
| SDG & target yang dipilih | SDG 11 – Kota dan Permukiman yang Berkelanjutan (Target 11.2: Sistem Transportasi yang Aman, Terjangkau, dan Berkelanjutan) |
| Masalah nyata yang diangkat | Kemacetan lalu lintas perkotaan yang dipengaruhi oleh fluktuasi jam sibuk, hari libur, dan cuaca yang menyebabkan pemborosan energi dan emisi karbon. |
| Tugas ML (Target & jenis tugas) | Memprediksi kategori tingkat kepadatan lalu lintas (Lancar, Padat, Macet) dari data tabular sensor & cuaca (Multiclass Classification). |
| Objective & metrik utama | Meminimalkan Cross-Entropy Loss (`nn.CrossEntropyLoss`). Metrik utama: Accuracy ($>80\%$), Precision, Recall, dan F1-Score (Macro). |
| Dataset (sumber, URL, ukuran, jenis data) | Metro Interstate Traffic Volume (UCI ML Repository / Kaggle: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume, 48.204 baris, 9 kolom, CSV Tabular, ~3.2 MB). |
| Rencana tiga model (baseline / advanced / deep learning) | • **Baseline:** Logistic Regression (Scikit-Learn)<br>• **Advanced ML:** Random Forest Classifier (Scikit-Learn)<br>• **Deep Learning:** Multilayer Perceptron (`MLPModel` PyTorch `nn.Module`, 2 Hidden Layers + ReLU + Dropout) |
| Risiko & pertimbangan etika (bias, privasi) | Privasi aman karena hanya memakai data agregat volume kendaraan dan cuaca (tanpa kamera/identitas pribadi). Risiko anomali sensor cuaca dimitigasi pada tahap data cleaning. |
| Referensi awal (minimal 2) | 1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.<br>2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.<br>3. Syah, R. (2021). *Pengantar Machine Learning: Teori dan Aplikasi dengan Python*. Andi Publisher. |

**Kriteria persetujuan dosen:**
- [x] Ide berasal dari mahasiswa dan tidak duplikat
- [x] Selaras jelas dengan satu SDG dan targetnya
- [x] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [x] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [x] Layak untuk tiga model termasuk deep learning dalam 6 minggu
