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
