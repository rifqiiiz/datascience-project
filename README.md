<div align="center">

# PREDIKSI TINGKAT KEPADATAN LALU LINTAS PERKOTAAN MENGGUNAKAN ALGORITMA KLASIFIKASI MACHINE LEARNING DAN MULTILAYER PERCEPTRON (PYTORCH) BERBASIS DATA SENSOR JALAN RAYA

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20PyTorch-orange)](https://pytorch.org/)
[![SDG 11](https://img.shields.io/badge/SDG-11%3A%20Sustainable%20Cities-FD9D24)](https://sdgs.un.org/goals/goal11)

</div>

Repositori ini memuat berkas proposal, perancangan pembelajaran mesin berbasis data tabular, serta dokumentasi eksperimen model klasifikasi tingkat kepadatan lalu lintas (*Lancar*, *Padat*, *Macet*) berdasarkan parameter sensor jalan raya, waktu, dan kondisi cuaca untuk mendukung **SDG 11 (Target 11.2)**. Seluruh pengembangan disusun mengacu pada siklus **L-M-O-P** (*Learning Problem, Model Design, Optimization, Predict & Evaluate*) dan selaras dengan materi perkuliahan **Data Science (TRPL22406)** – D4 Teknologi Rekayasa Perangkat Lunak, Politeknik Negeri Madiun.

## DAFTAR PUSTAKA DAN BERKAS DOKUMEN

Struktur direktori dokumentasi pada folder `docs/` dirinci sebagai berikut:

* [09_LAMPIRAN_B.md](09_LAMPIRAN_B.md)
  Formulir pengajuan ide checkpoint 1 yang memuat rumusan target klasifikasi multikelas tabular, daftar tautan dataset UCI/Kaggle, rencana tiga model (tanpa CNN), mitigasi risiko etika, dan referensi ilmiah.
* [09_minggu9.md](09_minggu9.md)
  Panduan capaian tugas dan kriteria checkpoint tahap awal (Minggu 1 & Minggu 2) untuk perumusan masalah pembelajaran serta penyiapan data bebas *data leakage*.
* [Laporan_Proyek_ML_LMOP.md](Laporan_Proyek_ML_LMOP.md)
  Draf utama template laporan komprehensif dari Bab 1 hingga Bab 10 yang mencakup seluruh alur siklus hidup proyek secara ilmiah menggunakan Scikit-Learn dan PyTorch.

---

## NOTEBOOK EKSPERIMEN

Struktur direktori kode dan notebook pada folder `notebooks/` dirinci sebagai berikut:

* [notebooks/01_learning_problem_eda.ipynb](01_learning_problem_eda.ipynb)
  Eksplorasi data awal (EDA), pembentukan kelas target tingkat kepadatan lalu lintas (*Lancar*, *Padat*, *Macet*), pengecekan kondisi dataset, serta visualisasi pola jam sibuk dan korelasi fitur.
* [notebooks/02_data_preparation.ipynb](02_data_preparation.ipynb)
  Pembersihan data (*data cleaning*), rekayasa fitur siklikal waktu (*feature engineering*), pembagian dataset secara *stratified* (60% Train, 20% Val, 20% Test), serta transformasi fitur bebas *data leakage*.

---

## RINGKASAN PROYEK

Proyek riset ini bertujuan memitigasi dampak kemacetan lalu lintas perkotaan melalui klasifikasi tingkat kepadatan jalan raya berbasis data tabular sensor dan meteorologi. Pendekatan komputasi memanfaatkan tiga tingkatan kompleksitas model secara bertahap:

* **Baseline Model**
  *Logistic Regression* (Scikit-Learn) dengan `StandardScaler` dan `OneHotEncoder` untuk mengevaluasi batas keputusan linier dasar antara fitur temporal dan cuaca terhadap tingkat kepadatan jalan.
* **Advanced Model**
  *Random Forest Classifier* (Scikit-Learn) untuk memetakan interaksi non-linier dan aturan ambang batas kompleks antar-variabel jam sibuk, hari libur, serta kondisi cuaca ekstrem.
* **Deep Learning Model**
  *Multilayer Perceptron* (`MLPModel` subclassing `torch.nn.Module` di **PyTorch**) khusus data tabular dengan arsitektur *Linear* multi-layer menggunakan aktivasi **ReLU**, `Dropout`, `nn.CrossEntropyLoss()`, serta *mini-batch* `DataLoader` dengan *optimizer* **Adam**.
