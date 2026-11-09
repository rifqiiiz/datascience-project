## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | Muhammad Rifqi Maulana / 244311018 <br/> Vicky Rizkyanto / 244311030 |
| Judul ide | Klasifikasi Tingkat Kepadatan Arus Lalu Lintas Perkotaan Menggunakan Convolutional Neural Network (CNN) dan Transfer Learning MobileNetV2 Berbasis Citra CCTV |
| SDG & target yang dipilih | SDG 11 – Kota dan Permukiman yang Berkelanjutan (Target 11.2: Akses Transportasi Berkelanjutan) |
| Masalah nyata yang diangkat | Kemacetan lalu lintas perkotaan yang meningkatkan waktu tempuh, emisi gas rumah kaca, dan pemborosan energi kendaraan. |
| Tugas ML (Target & jenis tugas) | Memprediksi kategori kepadatan lalu lintas (Sparse, Dense, Traffic Jam) berbasis input citra visual (Multiclass Image Classification). |
| Objective & metrik utama | Meminimalkan Categorical Cross-Entropy Loss. Metrik utama: Akurasi ($>85\%$), Precision, Recall, dan F1-Score Macro. |
| Dataset (sumber, URL, ukuran, jenis data) | Kaggle Traffic-Net Dataset (https://www.kaggle.com/datasets/umairshahpirzada/traffic-net, ~4,400 citra RGB 224x224, format JPG/PNG, ukuran ~200 MB). |
| Rencana tiga model (baseline / advanced / deep learning) | • **Baseline:** Logistic Regression<br>• **Advanced ML:** Random Forest + Fitur HOG (Histogram of Oriented Gradients)<br>• **Deep Learning:** CNN MobileNetV2 (Transfer Learning) |
| Risiko & pertimbangan etika (bias, privasi) | Risiko penurunan performa pada cuaca ekstrem. Privasi aman karena memproses tingkat penumpukan kendaraan secara agregat tanpa mengekstrak plat nomor atau identitas personal. |
| Referensi awal (minimal 2) | 1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.<br>2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. |

**Kriteria persetujuan dosen:**
- [x] Ide berasal dari mahasiswa dan tidak duplikat
- [x] Selaras jelas dengan satu SDG dan targetnya
- [x] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [x] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [x] Layak untuk tiga model termasuk deep learning dalam 6 minggu