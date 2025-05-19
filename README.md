# Laporan Proyek Chocolate Bar Rating Classification - Khoirotun Hisan

## Domain Proyek

Cokelat merupakan salah satu produk pangan yang digemari secara global. Banyak perusahaan cokelat bersaing menghasilkan produk berkualitas tinggi. Rating atau penilaian terhadap cokelat batangan dapat membantu konsumen memilih produk yang sesuai selera dan juga memberi wawasan bagi produsen dalam meningkatkan kualitas produknya.

Namun, penilaian terhadap cokelat bersifat subjektif dan dipengaruhi oleh berbagai fitur seperti lokasi perusahaan, kandungan kakao, dan asal biji kakao. Dengan adanya data historis, machine learning dapat dimanfaatkan untuk memprediksi kategori rating dari cokelat berdasarkan fitur-fitur tersebut.

**Mengapa perlu diselesaikan?**
Prediksi ini berguna untuk:

* Membantu produsen memahami faktor penting yang memengaruhi kualitas menurut persepsi konsumen.
* Memberikan estimasi rating produk baru sebelum dirilis ke pasar.

## Business Understanding

### Problem Statements

1. Bagaimana mengkategorikan rating cokelat batangan menjadi kelas low, medium, dan high?
2. Apakah fitur seperti persentase kakao, asal perusahaan, dan asal biji kakao dapat digunakan untuk memprediksi kategori rating?
3. Algoritma machine learning mana yang paling sesuai untuk memodelkan prediksi ini?

### Goals

1. Membuat klasifikasi multikelas untuk rating cokelat (low, medium, high).
2. Menggunakan fitur-fitur seperti lokasi perusahaan dan persentase kakao untuk pelatihan model.
3. Melakukan evaluasi model berdasarkan metrik klasifikasi seperti precision, recall, dan F1-score.

### Solution Statements

* Menggunakan Random Forest Classifier sebagai baseline model.
* Melakukan hyperparameter tuning untuk meningkatkan performa model.
* Menggunakan evaluasi berdasarkan classification report dan confusion matrix untuk menilai performa model.

## Data Understanding

Dataset yang digunakan berasal dari [Flavors of Cacao](https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings). Dataset ini berisi ulasan terhadap berbagai cokelat batangan dari berbagai perusahaan di seluruh dunia.

### Variabel-variabel pada dataset:

* **Company (Maker-if known)**: Nama produsen cokelat.
* **Specific Bean Origin (or Name)**: Lokasi asal spesifik dari biji kakao.
* **REF**: ID referensi.
* **Review Date**: Tahun review dilakukan.
* **Cocoa Percent**: Persentase kandungan kakao.
* **Company Location**: Negara tempat perusahaan berada.
* **Rating**: Rating dari 1.0–5.0.
* **Bean Type**: Jenis biji kakao (jika diketahui).
* **Broad Bean Origin**: Asal umum biji kakao.

EDA menunjukkan rating sebagian besar berada di kisaran 2.5–3.5 dan cocoa percent umum berada di 70%.

## Data Preparation

### Teknik Data Preparation yang Dilakukan:

1. Menghapus kolom tidak relevan: `Company (Maker-if known)` dan `Specific Bean Origin (or Name)`
2. Membersihkan dan menstandarkan nama kolom.
3. Mengubah `Cocoa Percent` dari string ke numerik.
4. Membuat label klasifikasi baru dari `Rating` menggunakan binning:

   * Low: rating < 3.0
   * Medium: 3.0 ≤ rating < 3.5
   * High: rating ≥ 3.5
5. Melakukan label encoding pada `Company Location` dan `Broad Bean Origin`.
6. Split data menjadi training (80%) dan testing (20%).

### Alasan Tahapan:

* Penghapusan kolom spesifik bertujuan menghindari overfitting dan noise.
* Encoding dan transformasi numerik dibutuhkan agar model bisa mengolah data kategorikal dan string.
* Binning rating menjadi kategori klasifikasi agar masalah dapat ditangani sebagai classification.

## Modeling

### Algoritma yang Digunakan:

* **Random Forest Classifier**

Model dilatih menggunakan `RandomForestClassifier` dari scikit-learn dengan parameter default terlebih dahulu, kemudian dilakukan hyperparameter tuning menggunakan `GridSearchCV`.

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
```

### Kelebihan dan Kekurangan Random Forest:

**Kelebihan:**

* Tahan terhadap overfitting.
* Mendukung fitur numerik dan kategorikal.
* Menyediakan feature importance.

**Kekurangan:**

* Model kompleks dan lebih sulit diinterpretasikan.
* Konsumsi memori tinggi saat banyak pohon digunakan.

### Improvement:

Model ditingkatkan dengan tuning hyperparameter dan ditemukan konfigurasi terbaik melalui GridSearchCV.

## Evaluation

### Metrik Evaluasi:

* **Accuracy**: Proporsi prediksi yang benar.
* **Precision**: Seberapa akurat model memprediksi kelas tertentu.
* **Recall**: Kemampuan model menemukan semua contoh dari suatu kelas.
* **F1-score**: Rata-rata harmonik precision dan recall.
* **Confusion Matrix**: Tabel evaluasi untuk melihat kesalahan klasifikasi antar kelas.

### Hasil Evaluasi:

```
              precision    recall  f1-score   support

        high       0.49      0.56      0.53       140
         low       0.45      0.28      0.34        90
      medium       0.41      0.46      0.43       129

    accuracy                           0.45       359
   macro avg       0.45      0.43      0.43       359
weighted avg       0.45      0.45      0.45       359
```

### Penjelasan:

* Model cukup baik dalam memprediksi kelas high, tetapi kesulitan membedakan antara low dan medium.
* Precision dan recall rendah untuk kelas low menunjukkan ketidakseimbangan atau keterbatasan informasi dalam fitur.
* F1-score sebagai indikator keseluruhan performa menunjukkan adanya ruang untuk perbaikan.

---

**Catatan Tambahan**:

* Dataset ini memiliki kemungkinan bias dan ketidakseimbangan distribusi kelas.
* Potensi perbaikan ke depan: menggunakan model boosting seperti XGBoost atau melakukan feature engineering lebih dalam.

---
