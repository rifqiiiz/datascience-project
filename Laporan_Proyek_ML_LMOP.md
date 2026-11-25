**INFORMASI PROYEK**

**Judul Proyek:**  
Prediksi Tingkat Kepadatan Lalu Lintas Perkotaan Menggunakan Algoritma Klasifikasi Machine Learning dan Multilayer Perceptron (PyTorch) Berbasis Data Sensor Jalan Raya

**Nama Mahasiswa:** Muhammad Rifqi Maulana / 244311018 - Vicky Rizkyanto / 244311030  
**NIM:** 244311018 & 244311030  
**Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak – Politeknik Negeri Madiun  
**Mata Kuliah:** Data Science (TRPL22406)  
**Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.  
**Tahun Akademik:** 2026 / Semester 5  
**Link GitHub Repository:** https://github.com/rifqiiiz/datascience-project  
**Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
**SDG yang Dipilih:** SDG 11 – Kota dan Permukiman yang Berkelanjutan  
**Target SDG yang Relevan:** Target 11.2 – Menyediakan akses ke sistem transportasi yang aman, terjangkau, mudah diakses, dan berkelanjutan bagi semua

---

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti kerangka kerja **ML Lifecycle (L-M-O-P)** yang terdiri dari empat tahap berulang (iteratif) sesuai kaidah *Dasar-dasar Machine Learning*:

| Tahap | Singkatan | Inti Pertanyaan | Bagian Laporan |
| :--- | :--- | :--- | :--- |
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

## 2. PROJECT OVERVIEW **[L]**

### 2.1 Latar Belakang

Kemacetan lalu lintas di area perkotaan merupakan salah satu tantangan transportasi terbesar yang berdampak langsung pada pemborosan bahan bakar, penurunan produktivitas masyarakat, serta peningkatan emisi gas rumah kaca. Manajemen lalu lintas konvensional yang hanya mengandalkan pengaturan jadwal tetap (*fixed-time traffic signals*) sering kali gagal beradaptasi dengan lonjakan volume kendaraan yang dipengaruhi oleh faktor jam sibuk, hari kerja/libur, serta perubahan kondisi cuaca.

Dengan memanfaatkan rekaman data sensor lalu lintas per jam (*traffic volume*) yang dipadukan dengan parameter meteorologi (suhu, curah hujan, tutupan awan, kondisi cuaca) dan atribut waktu, algoritma *Machine Learning* untuk data tabular mampu memodelkan pola kepadatan jalan raya secara akurat. Sistem ini mengklasifikasikan kondisi arus jalan menjadi tiga tingkat: *Lancar* ($< 2.500$ kendaraan/jam), *Padat* ($2.500 - 4.999$ kendaraan/jam), dan *Macet* ($\ge 5.000$ kendaraan/jam). Informasi prediksi ini dapat dimanfaatkan oleh dinas perhubungan untuk rekayasa lalu lintas maupun sistem navigasi publik.

**Referensi:**
1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. Syah, R. (2021). *Pengantar Machine Learning: Teori dan Aplikasi dengan Python*. Andi Publisher.
4. Hogue, J. (2019). *Metro Interstate Traffic Volume Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B

### 2.2 Keterkaitan dengan SDGs (WAJIB)

Ide proyek dirancang untuk mendukung **SDG 11: Kota dan Permukiman yang Berkelanjutan**.

| Komponen | Pertanyaan Panduan | Jawaban Mahasiswa |
| :--- | :--- | :--- |
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? | SDG 11 – Kota dan Permukiman yang Berkelanjutan |
| **Target SDG** | Target mana yang paling relevan? | Target 11.2 – Menyediakan akses ke sistem transportasi yang aman, terjangkau, mudah diakses, dan berkelanjutan bagi semua |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Kemacetan kronis pada koridor jalan utama perkotaan akibat fluktuasi volume kendaraan harian dan gangguan cuaca yang merugikan komuter, angkutan umum, serta meningkatkan polusi udara. |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan yang dibantu? | Algoritma ML dan Jaringan Saraf Tiruan (MLP) tabular mampu mempelajari pola interaksi non-linier antara waktu, hari libur, dan kondisi cuaca untuk memprediksi kepadatan lalu lintas secara cepat dan ringan tanpa memerlukan perangkat keras komputasi berat. |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG? | Memberikan prediksi dini tingkat kepadatan jalan untuk penyesuaian durasi lampu lalu lintas adaptif dan rekomendasi rute alternatif bagi pengguna transportasi perkotaan. |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? | Privasi pengguna jalan terlindungi sepenuhnya karena model hanya memproses data agregat sensor jalan dan cuaca (tanpa kamera/identitas pribadi). Risiko bias cuaca ekstrem dimitigasi melalui validasi stratifikasi. |

**Sumber ide:** Pengamatan langsung terhadap pola kepadatan lalu lintas perkotaan yang sangat dipengaruhi oleh jam berangkat/pulang kerja, akhir pekan, serta kondisi hujan, yang dapat dimodelkan secara efisien menggunakan data tabular sensor dan cuaca.

### 2.3 Ringkasan Alur Lifecycle Proyek

Proyek dimulai dengan perumusan masalah klasifikasi multikelas berbasis data tabular (**L**), dilanjutkan dengan eksplorasi, pembersihan, rekayasa fitur waktu siklikal (`hour_sin`, `hour_cos`), encoding, penskalaan, dan pembagian data 60% Train / 20% Validation / 20% Test (**L**). Selanjutnya dirancang tiga model secara bertahap (**M**): Logistic Regression (baseline linier), Random Forest Classifier (advanced non-linier), dan Multilayer Perceptron / MLP berbasis PyTorch `nn.Module` (deep learning tabular). Ketiga model dilatih dan dioptimasi (**O**) menggunakan regularisasi, *RandomizedSearchCV*, dan *mini-batch gradient descent* dengan Adam. Terakhir, performa diuji sekali pada *test set* (**P**) untuk evaluasi generalisasi dan umpan balik iterasi *lifecycle*.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING **[L]**

### 3.1 Problem Statements

1. Bagaimana melakukan rekayasa fitur (*feature engineering*) siklikal dan transformasi pada data tabular sensor jalan raya dan cuaca agar siap diproses oleh algoritma pembelajaran mesin tanpa terjadi *data leakage*?
2. Bagaimana perbandingan performa model linier sederhana (*Logistic Regression*) sebagai tolok ukur dasar dibandingkan model berbasis pohon keputusan (*Random Forest*) dan jaringan saraf tiruan tabular (*Multilayer Perceptron* di PyTorch)?
3. Bagaimana mengoptimalkan hyperparameter, fungsi aktivasi non-linier, dan regularisasi model untuk mencapai keseimbangan antara *underfitting* dan *overfitting* (*sweet spot* generalisasi)?

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Mahasiswa |
| :--- | :--- | :--- |
| **Target** | Apa yang diprediksi? Jenis tugas? | Kategori tingkat kepadatan lalu lintas (*Lancar*, *Padat*, *Macet*) yang diturunkan dari ambang operasional volume kendaraan per jam ($< 2.500$, $2.500 - 4.999$, $\ge 5.000$). Jenis tugas: **Klasifikasi Multikelas (*Multiclass Classification*)**. |
| **Objective** | Apa yang dioptimalkan secara matematis? Metrik utama? | Meminimalkan fungsi kerugian **Categorical Cross-Entropy (`nn.CrossEntropyLoss`)** saat pelatihan. Metrik evaluasi keberhasilan: **Accuracy** $> 80\%$ serta **Precision, Recall, dan F1-Score (Macro)** pada *validation/test set*. |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data? | Dataset *Metro Interstate Traffic Volume* (UCI ML Repository / Kaggle), 48.204 baris mentah (48.177 baris bersih) data tabular CSV dengan fitur numerik cuaca, waktu siklikal, dan kategorikal. Diasumsikan sampel dapat dibagi secara terstratifikasi untuk mengukur kemampuan generalisasi. |

### 3.3 Goals

1. Membangun pipeline pemrosesan data tabular yang bersih, terbebas dari *data leakage*, dan dapat direproduksi menggunakan Scikit-Learn.
2. Mengevaluasi dan membandingkan tiga tingkat kompleksitas model (Baseline Linier, Advanced Ensemble, dan Deep Learning MLP PyTorch) secara objektif.
3. Mencapai akurasi pengujian di atas 80% dengan selisih (*gap*) akurasi latih dan validasi yang kecil guna mendukung sistem perencanaan transportasi berkelanjutan (SDG 11.2).
4. Menyediakan repositori kode yang terdokumentasi lengkap dengan *environment* standar yang mudah diuji ulang.

### 3.4 Solution Approach **[M]**

Mengikuti prinsip pemilihan keluarga model secara bertahap (mulai dari model linier dengan rekayasa fitur yang baik, lalu menaikkan kompleksitas secara bertahap ke pohon keputusan/ensemble dan jaringan saraf tiruan):

#### **Model 1 – Baseline Model**
* **Algoritma:** Logistic Regression (Multinomial - Scikit-Learn)
* **Pendekatan Fitur:** 10 fitur numerik terstandarisasi (`StandardScaler`) dan 2 fitur kategorikal hasil `OneHotEncoder` (total 33 fitur input).
* **Justifikasi:** Model linier yang cepat dilatih, mudah ditafsirkan, dan berfungsi sebagai tolok ukur (*baseline*) yang jujur untuk melihat sejauh mana batas keputusan linier dapat memisahkan tingkat kepadatan jalan.

#### **Model 2 – Advanced / ML Model**
* **Algoritma:** Random Forest Classifier (Scikit-Learn)
* **Pendekatan Fitur:** Kombinasi fitur meteorologi, fitur rekayasa waktu siklikal (`hour_sin`, `hour_cos`), indikator domain (`is_weekend`, `is_rush_hour`), serta variabel kategorikal cuaca dan hari libur.
* **Justifikasi:** Model ensemble berbasis pohon keputusan non-linier yang mampu menangkap interaksi kompleks (misalnya: pengaruh hujan pada jam sibuk hari kerja dibandingkan pada hari libur) serta tahan terhadap pencilan (*outliers*).

#### **Model 3 – Deep Learning Model (WAJIB)**
* **Algoritma:** Multilayer Perceptron (`MLPModel` berbasis `torch.nn.Module` di **PyTorch**)
* **Arsitektur:** Minimal 2 *hidden layers* (`nn.Linear(33, 64)` $\rightarrow$ `nn.Linear(64, 32)`) dengan aktivasi non-linier **ReLU** ($f(x) = \max(0, x)$) dan `nn.Dropout(0.2)`.
* **Justifikasi:** Model linier memiliki keterbatasan mendasar pada data yang tidak terpisah secara linier (*non-linearly separable*, seperti analogi gerbang XOR). Berdasarkan *Universal Approximation Theorem* dan efisiensi jaringan berlapis (*deep*), MLP mempelajari representasi fitur non-linier langsung dari data (*data-dependent basis*) khusus untuk data tabular tanpa menggunakan CNN.

---

## 4. DATA UNDERSTANDING **[L]**

### 4.1 Informasi Dataset

* **Sumber Dataset:** UCI Machine Learning Repository / Kaggle – *Metro Interstate Traffic Volume* (https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
* **Jumlah Baris (Rows):** 48.204 observasi mentah (48.177 observasi bersih setelah pembersihan)
* **Jumlah Kolom (Columns/Features):** 9 kolom asli (dikembangkan menjadi 12 fitur terpilih dan 33 dimensi fitur setelah *One-Hot Encoding*)
* **Tipe Data:** Tabular (Numerik Kontinu, Diskret, Kategorikal, dan Datetime)
* **Ukuran Berkas:** ~3.2 MB
* **Format Berkas:** CSV (`.csv` / `.csv.gz`)

### 4.2 Deskripsi Fitur

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
| :--- | :--- | :--- | :--- |
| `holiday` | Categorical | Status hari libur nasional AS atau hari biasa (`None`) | `"None"`, `"Labor Day"`, `"Thanksgiving Day"` |
| `temp` | Float | Suhu rata-rata udara dalam satuan Kelvin | `288.28`, `295.14`, `265.40` |
| `rain_1h` | Float | Curah hujan dalam milimeter pada 1 jam terakhir | `0.0`, `1.25`, `5.40` |
| `snow_1h` | Float | Curah salju dalam milimeter pada 1 jam terakhir | `0.0`, `0.51` |
| `clouds_all` | Integer | Persentase tutupan awan di langit ($0\% - 100\%$) | `40`, `75`, `90` |
| `weather_main` | Categorical | Deskripsi kategori cuaca utama | `"Clear"`, `"Clouds"`, `"Rain"`, `"Mist"` |
| `weather_description` | Categorical | Deskripsi detail kondisi cuaca | `"sky is clear"`, `"light rain"`, `"broken clouds"` |
| `date_time` | Datetime | Stempel waktu pencatatan sensor dalam format waktu lokal | `"2012-10-02 09:00:00"` |
| `traffic_volume` | Integer | Jumlah kendaraan per jam (dikonversi menjadi label `target_label`) | `5545` (Macet), `3100` (Padat), `850` (Lancar) |

### 4.3 Kondisi Data

* **Missing Values:** Terdapat 48.143 nilai kosong (`NaN`) pada kolom `holiday` saat dimuat oleh `pandas` karena string `"None"` otomatis diinterpretasikan sebagai nilai hilang (diisi kembali menjadi kategori `"None"`).
* **Duplicate Data:** Ditemukan 17 baris duplikat identik di seluruh kolom yang dihapus menggunakan `drop_duplicates()`.
* **Outliers:** Terdapat pencilan fisik pada fitur `temp` bernilai $0\text{ K}$ (kesalahan bacaan sensor suhu di bawah $200\text{ K}$, sebanyak 10 baris).
* **Imbalanced Data:** Dengan membagi `traffic_volume` berdasarkan ambang batas operasional jalan raya ($< 2.500$: Lancar, $2.500 - 4.999$: Padat, $\ge 5.000$: Macet), proporsi ketiga kelas pada keseluruhan dataset bersih terdistribusi cukup merata (**36.32% Lancar**, **40.12% Padat**, **23.56% Macet**).
* **Noise:** Terdapat variasi deskripsi teks yang tumpang tindih pada kolom `weather_description` (38 sub-kategori) yang memiliki informasi identik dengan `weather_main` (11 kategori utama).
* **Asumsi Data:** Ekstraksi informasi waktu siklikal (`hour_sin`, `hour_cos`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`) menjadikan setiap baris observasi dapat diperlakukan sebagai sampel independen dan terdistribusi identik (*i.i.d.*) untuk tugas klasifikasi tabular.

### 4.4 Exploratory Data Analysis (EDA)

#### Visualisasi 1: Distribusi Kelas Target (`traffic_density`)
```
[Lancar (< 2,500)      ] ██████████████████   17,498 (36.32%)
[Padat  (2,500 - 4,999)] ████████████████████ 19,329 (40.12%)
[Macet  (>= 5,000)     ] ████████████         11,350 (23.56%)
```
**Insight:** Distribusi ketiga kelas target cukup berimbang tanpa kelas minoritas ekstrem sehingga metrik *Accuracy* dan *Macro F1-Score* dapat digunakan secara objektif.

#### Visualisasi 2: Pola Kepadatan Berdasarkan Jam (`hour`) dan Akhir Pekan (`is_weekend`)
```
Rata-rata Volume Kendaraan per Jam:
Hari Kerja (Sen-Jum) : Puncak ganda pada pukul 07:00-09:00 (~5,400) & 16:00-18:00 (~5,800)
Akhir Pekan (Sab-Min): Puncak tunggal landai pada pukul 12:00-15:00 (~4,100)
```
**Insight:** Interaksi antara atribut jam (`hour`) dan status akhir pekan (`is_weekend`) merupakan pembeda utama (*key driver*) antara kondisi arus lancar dan macet.

#### Visualisasi 3: Deteksi Pencilan (`Outliers`) dan Korelasi Antarfitur
```
Statistik Fitur Numerik Sebelum Pembersihan:
- temp    : Min = 0.0 K (anomali sensor), Median = 282.45 K, Max = 310.07 K
- rain_1h : Min = 0.0 mm, P99 = 2.54 mm, Max = 9831.3 mm
```
**Insight:** Nilai `temp < 200 K` ($0.0\text{ K}$) merupakan anomali sensor yang wajib dibersihkan sebelum penskalaan fitur (`StandardScaler`) agar nilai rata-rata dan standar deviasi tidak terdistorsi.

---

## 5. DATA PREPARATION **[L]**

### 5.1 Data Cleaning

* **Penghapusan Duplikat:** Menghapus 17 baris duplikat identik menggunakan `df.drop_duplicates()`.
* **Penanganan Outliers Sensor:** Menghapus 10 baris anomali suhu fisik tidak valid dengan filter `df[df["temp"] > 200]`, menyisakan **48.177 baris bersih**.
* **Penanganan Nilai Hilang `holiday`:** Mengisi nilai `NaN` pada kolom `holiday` dengan string `"None"` (`df["holiday"].fillna("None")`).
* **Diskretisasi Variabel Target:** Mengonversi `traffic_volume` menjadi kolom target `target_label` dengan aturan: `"Lancar"` ($< 2.500$), `"Padat"` ($2.500 - 4.999$), dan `"Macet"` ($\ge 5.000$), lalu di-encode menggunakan `LabelEncoder` (`0: Lancar`, `1: Macet`, `2: Padat`).

### 5.2 Feature Engineering

* **Ekstraksi Fitur Temporal & Siklikal dari `date_time`:**
  * `hour_sin` dan `hour_cos`: Transformasi trigonometri siklikal ($\sin(2\pi \cdot \text{hour}/24)$ dan $\cos(2\pi \cdot \text{hour}/24)$) agar model memahami bahwa pukul 23:00 berdekatan secara kontinu dengan pukul 00:00.
  * `day_of_week`: Indeks hari ($0 = \text{Senin}$ hingga $6 = \text{Minggu}$).
  * `month`: Bulan ($1 - 12$) untuk menangkap variasi musiman.
  * `is_weekend`: Fitur biner ($1$ jika Sabtu/Minggu, $0$ jika Senin–Jumat).
  * `is_rush_hour`: Fitur domain spesifik bernilai $1$ jika hari kerja pada rentang jam berangkat ($07.00 - 09.00$) atau jam pulang kerja ($16.00 - 18.00$).
* **Feature Selection:** Mempertahankan 10 fitur numerik (`temp`, `rain_1h`, `snow_1h`, `clouds_all`, `hour_sin`, `hour_cos`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`) dan 2 fitur kategorikal (`holiday`, `weather_main`), serta membuang `date_time`, `traffic_volume` (mencegah target leakage), dan `weather_description` (redundan).

### 5.3 Data Transformation

Menggunakan `ColumnTransformer` dari Scikit-Learn dengan pemisahan tipe fitur:
* **Fitur Numerik (10 kolom):** Ditransformasikan dengan `StandardScaler()` sehingga memiliki distribusi berpusat pada $\mu = 0$ dan $\sigma = 1$.
* **Fitur Kategorikal (`holiday` & `weather_main`):** Diubah menggunakan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` menghasilkan 23 kolom biner, sehingga total dimensi fitur input adalah **33 fitur**.

### 5.4 Data Splitting

Membagi dataset bersih (48.177 sampel) menjadi 3 bagian menggunakan `train_test_split` terstratifikasi dengan `random_state=42`:

```
Total Data Bersih : 48,177 sampel (33 fitur setelah transformasi)
├── Training Set   (60%): 28,905 sampel (melatih model & melakukan fit preprocessor)
├── Validation Set (20%):  9,636 sampel (penyetelan hyperparameter & arsitektur)
└── Test Set       (20%):  9,636 sampel (disimpan rapat & hanya dipakai SEKALI di akhir)
```

> **Pencegahan Kebocoran Data (Data Leakage):** Seluruh transformer (`StandardScaler` dan `OneHotEncoder`) **hanya di-fit pada `X_train`** (`preprocessor.fit_transform(X_train)`), kemudian diterapkan pada `X_val` dan `X_test` menggunakan `.transform()`.

### 5.5 Data Balancing

Tidak dilakukan teknik *oversampling* (seperti SMOTE) maupun *undersampling* karena ketiga kelas memiliki jumlah sampel yang memadai di setiap *split* (pada Validation Set: 3.500 Lancar, 3.866 Padat, 2.270 Macet) dan metrik evaluasi menggunakan *Macro F1-Score*.

### 5.6 Ringkasan Data Preparation (Apa – Mengapa – Bagaimana)

| Langkah | **Apa** yang Dilakukan | **Mengapa** Penting | **Bagaimana** Implementasinya |
| :--- | :--- | :--- | :--- |
| **1. Data Cleaning** | Menghapus 17 duplikat, membuang 10 anomali suhu $< 200\text{ K}$, dan mengisi `holiday` kosong. | Mencegah distorsi parameter rata-rata dan standar deviasi saat penskalaan fitur. | `drop_duplicates()`, filter `df["temp"] > 200`, dan `fillna("None")`. |
| **2. Feature Engineering** | Membuat fitur siklikal `hour_sin`, `hour_cos`, serta `is_weekend` dan `is_rush_hour`. | Merepresentasikan siklus waktu 24 jam secara kontinu dan memberi sinyal jam sibuk. | Transformasi $\sin/\cos$ pada `hour` dan indikator biner hari kerja. |
| **3. Data Splitting** | Membagi data menjadi 60% Train (28.905), 20% Val (9.636), dan 20% Test (9.636). | Menjamin evaluasi generalisasi yang jujur pada data yang belum pernah dilihat model. | Dua tahap `train_test_split(..., stratify=y, random_state=42)`. |
| **4. Transformation** | Standarisasi 10 fitur numerik dan *One-Hot Encoding* 2 fitur kategorikal (33 dimensi). | Menyamakan skala fitur kontinu dan mencegah bias urutan pada kategori cuaca/libur. | `ColumnTransformer` dengan `fit` khusus pada `X_train` saja. |

---

## 6. MODELING **[M + O]**

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model **[M]**
* **Nama Model:** Logistic Regression (Multinomial)
* **Teori Singkat:** Menghitung kombinasi linier dari fitur input ($z = W x + b$) untuk setiap kelas dan mengubahnya menjadi probabilitas melalui fungsi *Softmax*.
* **Alasan Pemilihan:** Menjadi tolok ukur dasar (*baseline*) yang cepat, deterministik, dan mudah dijelaskan sebelum beralih ke model non-linier yang lebih kompleks.

#### 6.1.2 Hypothesis Space & Inductive Bias **[M]**
* **Model Family:** Generalized Linear Model (Multinomial Logistic Regression).
* **Hypothesis Space:** Himpunan seluruh bidang datar (*hyperplane*) pemisah linier $h_W(x) = \text{softmax}(W x + b)$ pada ruang 33 fitur hasil transformasi.
* **Inductive Bias:** Mengasumsikan bahwa log-odds setiap kelas kepadatan lalu lintas merupakan kombinasi linier dari fitur waktu siklikal dan cuaca, serta mengasumsikan batas keputusan antar-kelas dapat dipisahkan secara linier.
* **Implikasi terhadap Dataset:** Cukup baik dalam mengenali kondisi arus *Lancar* (F1 = 0.88), namun kesulitan memisahkan batas transisi non-linier antara kelas *Padat* (F1 = 0.69) dan *Macet* (F1 = 0.65).

#### 6.1.3 Implementasi (Ringkas) **[M]**
```python
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500, random_state=42)
model_baseline.fit(X_train, y_train)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi **[O]**
* **Objective/Loss:** Multinomial Cross-Entropy Loss dengan penalti regularisasi $L_2$.
* **Eksperimen Parameter `C` pada Validation Set:**

| `C` (Regularization) | Train Accuracy | Val Accuracy | Val F1-Macro | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| `0.01` | 0.7514 | 0.7493 | 0.7352 | 2.59 |
| `0.10` | 0.7534 | 0.7505 | 0.7386 | 1.27 |
| `1.00` | 0.7528 | 0.7509 | 0.7392 | 1.16 |
| **`10.00`** | **0.7528** | **0.7510** | **0.7393** | 4.64 |

#### 6.1.5 Hasil Awal **[O → P]**
* **Best Parameter:** `C = 10.0`, `solver = 'lbfgs'`, `max_iter = 500`
* **Akurasi Validation Set:** **75.10%** (`0.7510`) | **Macro F1-Score:** **0.7393** (Train Accuracy: 75.28% — menunjukkan model linier mengalami *underfitting* karena batas keputusan bersifat non-linier).

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model **[M]**
* **Nama Model:** Random Forest Classifier
* **Teori Singkat:** Algoritma *ensemble* yang melatih banyak pohon keputusan (*Decision Trees*) secara paralel pada subsampel data (*bootstrap aggregating*) dan mengambil keputusan akhir berdasarkan suara terbanyak (*majority voting*).
* **Alasan Pemilihan:** Sangat unggul untuk data tabular campuran (numerik dan kategorikal) dan mampu menangkap interaksi non-linier antarfitur secara otomatis.
* **Keunggulan:** Tahan terhadap *overfitting* dibanding pohon keputusan tunggal, tidak sensitif terhadap skala fitur, dan menyediakan skor *feature importance*.
* **Kelemahan:** Ukuran model lebih besar dan waktu penyetelan hyperparameter lebih lama dibandingkan model linier.

#### 6.2.2 Hypothesis Space & Inductive Bias **[M]**
* **Model Family:** Ensemble Decision Trees (Bagging).
* **Hypothesis Space:** Gabungan fungsi konstan sepotong-sepotong (*piecewise constant*) dari ratusan pohon keputusan yang mempartisi ruang fitur secara rekursif sejajar sumbu (*axis-aligned splits*).
* **Inductive Bias:** Data tabular dapat dipisahkan secara efektif menggunakan aturan ambang batas dan interaksi non-linier antarfitur (misalnya kombinasi jam sibuk pada hari kerja saat cuaca hujan) tanpa memerlukan hubungan linier global.
* **Perbandingan dengan Baseline:** Mampu melampaui keterbatasan garis lurus *Logistic Regression* dengan peningkatan akurasi validasi sebesar **+15.50%** (dari 75.10% menjadi 90.60%).

#### 6.2.3 Implementasi Awal & Perbandingan Validation Set **[M]**
```python
from sklearn.ensemble import RandomForestClassifier

rf_default = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_default.fit(X_train, y_train)
```

**Perbandingan Awal Baseline vs Advanced (Validation Set):**

| Model | Konfigurasi Awal | Train Accuracy | Val Accuracy | Val F1-Macro |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1 — Baseline (Logistic Regression)** | `C=10.0, solver='lbfgs'` | 75.28% | 75.10% | 0.7393 |
| **Model 2 — Advanced (Random Forest Default)** | `n_estimators=100, max_depth=None` | 99.99% | **89.77%** | **0.8951** |

#### 6.2.4 Optimization: Hyperparameter Tuning **[O]**
* **Metode:** `RandomizedSearchCV` (`n_iter=12`, `cv=3`, `scoring='f1_macro'`, `random_state=42`) pada data latih.
* **Search Space:** `n_estimators: [100, 200, 300]`, `max_depth: [10, 15, 20, None]`, `min_samples_split: [2, 5, 10]`, `min_samples_leaf: [1, 2, 4]`, `max_features: ['sqrt', 'log2']`.
* **Best Parameters:** `{'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2'}`.
* **Skor Validasi (Sebelum vs Sesudah Tuning):**
  * **Default RF:** Train Acc = `99.99%`, Val Acc = `89.77%`, Val F1-Macro = `0.8951` (terindikasi *overfitting* karena pohon tumbuh tanpa batas kedalaman).
  * **Tuned RF:** Train Acc = `95.58%`, Val Acc = **`90.60%`**, Val F1-Macro = **`0.9035`**.

#### 6.2.5 Kontrol Overfitting **[O]**
Menerapkan pembatasan kedalaman maksimum pohon (`max_depth=20`), jumlah minimum sampel untuk membelah simpul (`min_samples_split=5`), dan pemilihan subset fitur logaritmik (`max_features='log2'`) sehingga *generalization gap* (Train vs Val) menyusut dari $10.22\%$ menjadi $4.98\%$ sekaligus menaikkan akurasi validasi.

#### 6.2.6 Hasil Model **[O → P]**
* **Akurasi Validation Set:** **90.60%** (`0.9060`) | **Macro F1-Score:** **0.9035** (Rincian per kelas pada Validation Set: Lancar F1 = `0.95`, Padat F1 = `0.89`, Macet F1 = `0.88`).

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model **[M]**
* **Nama Model:** Multilayer Perceptron (`MLPModel` berbasis `torch.nn.Module` di **PyTorch**)
* **Jenis Deep Learning:**
  * [x] Multilayer Perceptron (MLP) - untuk tabular
  * [ ] Convolutional Neural Network (CNN) - untuk image
  * [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
* **Alasan Pemilihan:** Model linier memiliki keterbatasan mendasar dalam memisahkan data yang tidak terpisah secara linier (*non-linearly separable*, seperti analogi gerbang logika XOR pada materi Pertemuan 12). Dengan menyusun beberapa lapisan tersembunyi (*hidden layers*) dan fungsi aktivasi non-linier **ReLU** ($f(x) = \max(0, x)$), MLP mampu melakukan aproksimasi fungsi kompleks (*Universal Approximation*) serta mempelajari *data-dependent basis* secara langsung dari data tabular sensor dan cuaca.

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias **[M]**
* **Inductive Bias Arsitektur:** *Fully-connected feedforward network* yang tidak mengasumsikan struktur spasial gambar (seperti CNN), melainkan memetakan kombinasi bobot global antar seluruh fitur tabular ($a_j^{(l)} = \sum w_{ji}^{(l)} z_i^{(l-1)} + w_{j0}^{(l)}$) yang dilewatkan ke fungsi aktivasi non-linier $\text{ReLU}(z) = \max(0, z)$ untuk mencegah *vanishing gradient*.
* **Hypothesis Space:** Himpunan seluruh fungsi komposisi non-linier berparameter bobot dan bias $h_W(x) = W^{(3)} \text{ReLU}(W^{(2)} \text{ReLU}(W^{(1)} x + b^{(1)}) + b^{(2)}) + b^{(3)}$ dengan arsitektur `[33, 64, 32, 3]`.

**Deskripsi Layer:**
```
1. Input Layer   : torch.tensor shape (33,), dtype=torch.float32
2. Hidden Layer 1: nn.Linear(33, 64) + F.relu + nn.Dropout(p=0.2)
3. Hidden Layer 2: nn.Linear(64, 32) + F.relu + nn.Dropout(p=0.2)
4. Output Layer  : nn.Linear(32, 3) -> Raw Logits (diproses langsung oleh nn.CrossEntropyLoss)
```

#### 6.3.3 Input & Preprocessing Khusus **[M]**
* **Input Shape:** Vektor tensor 1D berdimensi `(33,)` bertipe `torch.float32` (hasil transformasi 10 fitur numerik terstandarisasi dan 23 kolom biner *one-hot encoded*), serta label target bertipe `torch.long`.
* **Preprocessing Khusus:** Seluruh input numerik dipastikan memiliki skala seragam ($\mu=0, \sigma=1$) dan dikemas ke dalam `TensorDataset` serta `DataLoader` untuk pemrosesan *mini-batch*.

#### 6.3.4 Optimization: Konfigurasi Training **[O]**
```
- Framework      : PyTorch (torch.nn & torch.optim)
- Optimizer      : torch.optim.Adam
- Learning rate  : 0.001 (dibandingkan dengan 0.01)
- Loss function  : nn.CrossEntropyLoss() (menggabungkan LogSoftmax + NLLLoss)
- Batch size     : 64 (menggunakan DataLoader dengan shuffle=True)
- Epochs         : 30 epochs (memenuhi syarat minimal >= 10 epochs)
- Regularisasi   : Dropout (p=0.2) & Weight Decay (L2 penalty = 1e-4)
- Reproducibility: torch.manual_seed(42) & torch.Generator().manual_seed(42)
```

**Eksperimen Optimasi (Validation Set):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Exp-1 | 0.01 | 128 | Tanpa Dropout | 0.312 | 88.40% |
| Exp-2 | 0.001 | 64 | Dropout (0.2) + Weight Decay (1e-4) | **0.241** | **90.85%** |

#### 6.3.5 Implementasi (PyTorch `nn.Module` & Autograd Loop) **[M + O]**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 1. Konfigurasi Reproducibility & Device
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Konversi Data ke PyTorch Tensor & DataLoader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
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

input_dim = X_train.shape[1]  # 33 fitur
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

#### 6.3.6 Training Process & Diagnosis Optimasi **[O]**
* **Training Time:** ~16 detik untuk 30 epochs (pada Google Colab Standard Runtime).
* **Computational Resource:** CPU Intel Xeon / GPU T4 Google Colab.

**Analisis Training:**
* **Overfitting:** Tidak terjadi overfitting signifikan; selisih akurasi pelatihan (91.42%) dan validasi (90.85%) hanya sebesar $0.57\%$, menunjukkan titik seimbang (*sweet spot*) tercapai.
* **Konvergensi:** Nilai *validation loss* menurun stabil sejak epoch ke-1 hingga epoch ke-24 lalu mendatar (*converge*) tanpa lonjakan liar.
* **Learning Rate:** Nilai `0.001` dengan *Adam* menghasilkan pembaruan bobot yang mulus dan konsisten.

#### 6.3.7 Model Summary
```
MLPModel(
  (layers): ModuleList(
    (0): Linear(in_features=33, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=32, bias=True)
    (2): Linear(in_features=32, out_features=3, bias=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
)
Total Trainable Parameters: 4,355
```

---

## 7. EVALUATION **[P]**

### 7.1 Metrik Evaluasi

Untuk tugas klasifikasi multikelas tingkat kepadatan lalu lintas, digunakan metrik:
* **Accuracy:** Mengukur proporsi prediksi benar di seluruh kategori jalan raya.
* **Precision (Macro):** $\frac{TP}{TP + FP}$, memastikan sistem tidak sering memberikan peringatan macet palsu (*false alarm*).
* **Recall (Macro):** $\frac{TP}{TP + FN}$, memastikan kondisi kemacetan aktual dapat terdeteksi secara menyeluruh.
* **F1-Score (Macro):** Rata-rata harmonik dari Precision dan Recall untuk menjamin keseimbangan performa di ketiga kelas.

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Evaluasi dilakukan **sekali** pada 9.636 sampel *Test Set* yang tidak pernah dilibatkan selama proses pelatihan maupun penyetelan hyperparameter.

#### 7.2.1 Model 1 (Baseline — Logistic Regression, `C=10.0`)
```
- Accuracy  : 0.7502
- Precision : 0.7385
- Recall    : 0.7391
- F1-Score  : 0.7388
```

#### 7.2.2 Model 2 (Advanced/ML — Tuned Random Forest Classifier)
```
- Accuracy  : 0.9048
- Precision : 0.9029
- Recall    : 0.9021
- F1-Score  : 0.9025
```

**Top 5 Feature Importance (Tuned Random Forest):**
1. `num__hour_cos` (Komponen kosinus jam operasional): $41.1\%$
2. `num__hour_sin` (Komponen sinus jam operasional): $22.0\%$
3. `num__temp` (Suhu udara): $8.8\%$
4. `num__is_rush_hour` (Indikator jam sibuk): $7.9\%$
5. `num__day_of_week` (Hari dalam seminggu): $7.4\%$

#### 7.2.3 Model 3 (Deep Learning — PyTorch `MLPModel`)
```python
# Evaluasi akhir pada Test Set
model_dl.eval()
with torch.no_grad():
    X_te_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_dl = torch.argmax(model_dl(X_te_tensor), dim=1).cpu().numpy()
```
```
- Accuracy  : 0.9074
- Precision : 0.9056
- Recall    : 0.9049
- F1-Score  : 0.9052
```

**Confusion Matrix Model 3 (MLP PyTorch pada Test Set — 9.636 sampel):**
```
                 Pred: Lancar   Pred: Macet   Pred: Padat
Actual: Lancar       3,275           18            207
Actual: Macet            0        2,008            262
Actual: Padat          121          284          3,461
```

### 7.3 Perbandingan Ketiga Model

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Training Time | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Logistic Regression)** | 0.7502 | 0.7385 | 0.7391 | 0.7388 | 4.64 detik | 0.001 detik |
| **Advanced (Tuned Random Forest)** | 0.9048 | 0.9029 | 0.9021 | 0.9025 | 106.20 detik (tuning) | 0.045 detik |
| **Deep Learning (PyTorch MLP)** | **0.9074** | **0.9056** | **0.9049** | **0.9052** | 16.00 detik | 0.008 detik |

### 7.4 Analisis Hasil

1. **Model Terbaik:** Multilayer Perceptron berbasis PyTorch (`MLPModel`) mencapai performa tertinggi dengan akurasi pengujian **90.74%** dan Macro F1-Score **0.9052**, diikuti sangat ketat oleh Tuned Random Forest (**90.48%**).
2. **Perbandingan dengan Baseline:** Peningkatan akurasi sebesar **+15.72%** dari model linier (*Logistic Regression*, 75.02%) ke model non-linier (*Random Forest* dan *PyTorch MLP*) membuktikan secara empiris bahwa hubungan antara waktu siklikal, cuaca, dan kepadatan lalu lintas tidak terpisah secara linier.
3. **Trade-off:** Random Forest memberikan interpretabilitas tinggi melalui skor *feature importance* (`hour_cos` dan `hour_sin` menyumbang $>63\%$ kontribusi keputusan), sedangkan PyTorch MLP unggul dalam kecepatan pelatihan langsung (tanpa *grid/random search* pohon yang lama), kecepatan inferensi, dan akurasi generalisasi.
4. **Error Analysis:** Seperti terlihat pada *Confusion Matrix*, hampir tidak ada kesalahan ekstrem antara kelas *Lancar* dan *Macet* ($0$ sampel *Macet* diprediksi *Lancar*). Kesalahan prediksi terkonsentrasi pada batas transisi berdekatan antara kelas *Padat* dan *Macet* (misalnya volume kendaraan di kisaran $4.700 - 5.200$ kendaraan/jam) atau saat terjadi acara khusus yang tidak tercatat dalam data cuaca.
5. **Generalization Gap:** Performa pada *Validation Set* (90.85%) dan *Test Set* (90.74%) sangat konsisten (selisih hanya $0.11\%$), menandakan model memiliki kemampuan generalisasi yang sangat baik.

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

| Temuan dari Evaluasi [P] | Tahap yang Diperbaiki | Tindakan | Dampak pada Metrik |
| :--- | :--- | :--- | :--- |
| Model linier kesulitan memodelkan transisi jam 23:00 ke 00:00 secara kontinu | [L] Feature Engineering | Menambahkan fitur siklikal `hour_sin`, `hour_cos`, serta `is_weekend` dan `is_rush_hour` | Memberikan fitur utama bagi model non-linier ($>70\%$ total importance) |
| Random Forest default mengalami overfitting (Train 99.99%, Val 89.77%) | [O] Optimization | Penyetelan via `RandomizedSearchCV` (`max_depth=20`, `min_samples_split=5`, `max_features='log2'`) | Train Acc turun ke 95.58%, Val Acc naik ke **90.60%** |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

* **Model Terbaik:** Multilayer Perceptron berbasis PyTorch (`MLPModel`) untuk Data Tabular (dengan Tuned Random Forest sebagai alternatif kompetitif).
* **Alasan:** Mampu menangkap interaksi non-linier kompleks antara dimensi temporal siklikal dan parameter cuaca dengan generalisasi tinggi pada data uji.
* **Pencapaian Goals:** Seluruh tujuan tercapai, termasuk target akurasi pengujian $>80\%$ (tercapai **90.74%**) dan implementasi *pipeline* bebas *data leakage*.

### 8.2 Key Insights

* **Insight dari Data:** Faktor temporal (`hour_cos`, `hour_sin`, `is_rush_hour`, `day_of_week`, `is_weekend`) dan suhu (`temp`) menjadi pendorong utama prediksi kepadatan lalu lintas dibandingkan variabel curah hujan/salju harian.
* **Insight dari Modeling:** Pada data tabular yang telah melalui rekayasa fitur siklikal dan penskalaan yang tepat, model klasik (*Random Forest*) dan *MLP* sederhana di PyTorch melampaui model linier hingga $>15\%$ dan ringan dijalankan tanpa memerlukan CNN.

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap | Keputusan Kunci | Pelajaran |
| :--- | :--- | :--- |
| **L** – Learning Problem | Formulasi klasifikasi 3 kelas kepadatan ($<2.500$, $2.500-4.999$, $\ge 5.000$) dan fitur siklikal | Representasi siklikal waktu (`hour_sin`/`hour_cos`) sangat efektif membantu model memetakan siklus harian. |
| **M** – Model Design | Perbandingan bertahap: Linier $\rightarrow$ Tree Ensemble $\rightarrow$ PyTorch MLP | Pemilihan keluarga model non-linier krusial ketika batas keputusan antar-kelas tidak terpisah secara linier. |
| **O** – Optimization | Regularisasi `max_depth=20`, `Dropout(0.2)`, `Weight Decay`, dan `Adam` | Regularisasi efektif menekan *overfitting* pada pohon keputusan maupun jaringan saraf tiruan. |
| **P** – Predict & Evaluate | Evaluasi tunggal pada *Test Set* yang terisolasi | Menjaga aturan emas *train-val-test split* menjamin kejujuran skor generalisasi. |

### 8.4 Kontribusi Proyek

* **Manfaat Praktis:** Model siap digunakan sebagai modul prediksi kepadatan lalu lintas berbasis API ringan yang hanya membutuhkan masukan data waktu dan prakiraan cuaca lokal.
* **Kontribusi terhadap SDGs:** Mendukung **SDG 11.2** dengan menyediakan sistem pendukung keputusan untuk pengaturan sinyal lalu lintas adaptif dan perencanaan transportasi publik yang lebih efisien dan rendah emisi.

---

## 9. FUTURE WORK (Opsional)

**Data:**
* [x] Menambahkan fitur kalender acara lokal/konser dan data kecelakaan lalu lintas
* [x] Mengintegrasikan data sensor lalu lintas multi-koridor jalan di Indonesia

**Model:**
* [x] Mengeksplorasi algoritma *Gradient Boosting* modern seperti LightGBM dan CatBoost
* [x] Menguji pendekatan *Time-Series Forecasting* (lag features volume 1–3 jam sebelumnya)

**Deployment:**
* [x] Membuat REST API ringan menggunakan FastAPI
* [x] Membuat dasbor pemantauan interaktif menggunakan Streamlit

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

* **Link Repository:** https://github.com/rifqiiiz/datascience-project

**Struktur Folder:**
```
datascience-project/
├── data/                                      # [L] Dataset & hasil split terproses (.npz)
├── docs/                                      # Dokumentasi laporan & lampiran
│   ├── 09_LAMPIRAN_B.md
│   ├── 09_minggu9.md
│   └── Laporan_Proyek_ML_LMOP.md
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb          # [L] EDA & visualisasi distribusi
│   ├── 02_data_preparation.ipynb              # [L] Feature engineering siklikal & splitting
│   ├── 03_model_design_baseline_ml.ipynb      # [M+O] Model 1: Logistic Regression
│   ├── 03_model_design_advanced_ml.ipynb      # [M+O] Model 2: Random Forest + Tuning
│   ├── 04_deep_learning_optimization.ipynb    # [M+O] Model 3: PyTorch MLPModel
│   └── 05_evaluation_comparison.ipynb         # [P] Evaluasi akhir pada Test Set
├── models/                                    # Preprocessor & bobot model terlatih
├── requirements.txt                           # Daftar versi pustaka Python
└── README.md                                  # Dokumentasi utama repositori
```

### 10.2 Environment & Dependencies

* **Python Version:** 3.10.12
```text
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
torch==2.1.0
```

### 10.3 Reproducibility Checklist

* [x] Random seed ditetapkan (`random_state=42` dan `torch.manual_seed(42)`) di seluruh proses splitting dan pelatihan
* [x] Notebook dapat dijalankan ulang secara berurutan dari awal hingga akhir
* [x] Test set terisolasi penuh dan hanya digunakan sekali pada tahap evaluasi akhir
* [x] Versi pustaka tercatat lengkap di `requirements.txt`
