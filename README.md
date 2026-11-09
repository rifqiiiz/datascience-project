<div align="center">

# KLASIFIKASI TINGKAT KEPADATAN LALU LINTAS PERKOTAAN MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN) DAN TRANSFER LEARNING BERBASIS CITRA CCTV

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![SDG 11](https://img.shields.io/badge/SDG-11%3A%20Sustainable%20Cities-0A97D9)](https://sdgs.un.org/goals/goal11)

</div>

Repositori ini memuat berkas proposal, perancangan pembelajaran mesin, serta dokumentasi eksperimen model klasifikasi tingkat kepadatan lalu lintas (Lancar, Padat, Macet) berbasis pengolahan citra CCTV perkotaan. Seluruh pengembangan disusun mengacu pada siklus L-M-O-P (*Learning Problem, Model Design, Optimization, Predict & Evaluate*).

## DAFTAR PUSTAKA DAN BERKAS DOKUMEN

Struktur direktori dokumentasi pada folder `docs/` dirinci sebagai berikut:

- [docs/09_LAMPIRAN_B.md](docs/09_LAMPIRAN_B.md)
  Formulir pengajuan ide checkpoint 1 yang memuat rumusan target klasifikasi multikelas, daftar tautan dataset Kaggle, rencana tiga model, mitigasi risiko etika privasi, dan referensi ilmiah.
- [docs/09_minggu9.md](docs/09_minggu9.md)
  Panduan capaian tugas dan kriteria checkpoint tahap awal untuk perumusan masalah pembelajaran dan alur lifecycle L-M-O-P.
- [docs/Laporan_Proyek_ML_LMOP.md](docs/Laporan_Proyek_ML_LMOP.md)
  Draf utama template laporan komprehensif dari Bab 1 hingga Bab 10 yang mencakup seluruh alur siklus hidup proyek secara ilmiah.

---

## RINGKASAN PROYEK

Proyek riset ini bertujuan memitigasi dampak kemacetan lalu lintas perkotaan dan mendukung *Intelligent Transportation System* (ITS) melalui pengenalan tingkat kepadatan jalan raya secara otomatis berbasis vision. Pendekatan komputasi memanfaatkan tiga tingkatan model secara bertahap:

- Baseline Model
  *Logistic Regression* dengan masukan fitur warna/histogram sederhana untuk mengevaluasi ambang batas pemisahan kelas dasar.
- Advanced Model
  *Random Forest Classifier* dan *Support Vector Machine (SVM)* memanfaatkan ekstraksi fitur *Histogram of Oriented Gradients* (HOG) untuk memetakan tekstur dan orientasi bentuk kendaraan.
- Deep Learning Model
  *Convolutional Neural Network* (CNN Kustom) serta *Transfer Learning* (MobileNetV2 / ResNet50) dengan lapisan *Dropout*, *Batch Normalization*, dan pengoptimal Adam untuk mempelajari representasi fitur spasial citra secara mendalam.
