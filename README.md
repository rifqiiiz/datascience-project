KLASIFIKASI TINGKAT KEPADATAN LALU LINTAS PERKOTAAN MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN) DAN TRANSFER LEARNING BERBASIS CITRA CCTV

Repositori ini memuat berkas proposal, perancangan pembelajaran mesin, serta dokumentasi eksperimen model klasifikasi tingkat kepadatan arus lalu lintas berbasis citra CCTV perkotaan. Seluruh pengembangan disusun mengacu pada siklus L-M-O-P (Learning Problem, Model Design, Optimization, Predict & Evaluate).

DAFTAR PUSTAKA DAN BERKAS DOKUMEN

Struktur direktori dokumentasi pada folder docs/ dirinci sebagai berikut:

docs/09_LAMPIRAN_B.md
Formulir pengajuan ide checkpoint 1 yang memuat rumusan target klasifikasi, daftar tautan dataset, rencana tiga model, mitigasi risiko etika, dan referensi ilmiah.

docs/09_minggu9.md
Panduan capaian tugas dan kriteria checkpoint tahap awal untuk perumusan masalah pembelajaran.

docs/Laporan_Proyek_ML_LMOP.md
Draf utama template laporan komprehensif dari Bab 1 hingga Bab 10 yang mencakup seluruh alur siklus hidup proyek.

RINGKASAN PROYEK

Proyek riset ini bertujuan memitigasi dampak kemacetan dan mengoptimalkan manajemen transportasi perkotaan melalui pemantauan visual otomatis dari kamera CCTV. Pendekatan komputasi memanfaatkan tiga tingkatan model secara bertahap:

Baseline Model
Logistic Regression menggunakan nilai piksel RGB sederhana untuk mengevaluasi garis batas pemisahan linear dasar antar tingkat kepadatan.

Advanced Model
Random Forest Classifier dipadukan dengan ekstraksi fitur Histogram of Oriented Gradients (HOG) untuk memetakan kontur dan bentuk kendaraan secara non-linear.

Deep Learning Model
Convolutional Neural Network (CNN) berbasis Transfer Learning (MobileNetV2) dengan penambahan Global Average Pooling, aktivasi ReLU, Dropout, serta Softmax Classifier guna mengekstraksi representasi spasial hirarkis secara end-to-end.
