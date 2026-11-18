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
* **Missing Values:** Terdapat 48143 nilai kosong (`NaN`) pada Kolom `holiday`.
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
