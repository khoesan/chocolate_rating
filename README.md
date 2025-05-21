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

---

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari [Flavors of Cacao](https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings), yang memuat informasi mengenai berbagai cokelat batangan dari berbagai negara, termasuk rating rasa, produsen, asal biji kakao, dan tahun produksi.

### Jumlah Data

* Total **1.795 baris data** (sampel cokelat)
* Jumlah **9 kolom**

### Deskripsi Kolom

Berikut beberapa kolom penting dari dataset:

* `Company (Maker-if known)`: Nama produsen cokelat.
* `Specific Bean Origin (or Name)`: Lokasi asal spesifik dari biji kakao.
* `REF`: ID referensi.
* `Review Date`: Tahun review dilakukan.
* `Cocoa Percent`: Persentase kandungan kakao.
* `Company Location`: Negara tempat perusahaan berada.
* `Rating`: Rating dari 1.0–5.0.
* `Bean Type`: Jenis biji kakao (jika diketahui).
* `Broad Bean Origin`: Asal umum biji kakao.

### Kondisi Data

* **Missing values:**
  * `Bean Type`: 47 nilai hilang
  * `Broad Bean Origin`: 10 nilai hilang
* **Data duplikat:** Tidak ditemukan baris duplikat
* **Outlier:** Distribusi nilai `Rating` berkisar antara 1.0 hingga 5.0, dengan mayoritas berada di rentang 2.5 hingga 4.0. Beberapa produk memiliki rating ekstrem (di bawah 2.0 atau di atas 4.5) yang dapat dianggap sebagai outlier.

### Visualisasi Distribusi Rating

```python
plt.figure(figsize=(8, 4))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribusi Rating Cokelat')
plt.xlabel('Rating')
plt.ylabel('Jumlah Produk')
plt.show()
```

Grafik di atas menunjukkan bahwa sebagian besar produk cokelat memiliki rating antara 3.0 dan 4.0.
Data dibersihkan dan dikategorikan ulang pada kolom `Rating` untuk menghasilkan label klasifikasi: `low`, `medium`, dan `high`.
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

---

## Modeling

### Algoritma yang Digunakan

Model yang digunakan dalam proyek ini adalah **Random Forest Classifier** dari pustaka `scikit-learn`. Random Forest adalah algoritma ensemble berbasis pohon keputusan yang menggabungkan beberapa model Decision Tree untuk meningkatkan akurasi dan mengurangi risiko overfitting.

### Cara Kerja Random Forest (Konseptual)

1. **Bootstrap Sampling (Bagging)**
   Model membuat beberapa *subset* dari data latih menggunakan pengambilan sampel acak dengan pengembalian.

2. **Pembuatan Pohon Keputusan (Decision Trees)**
   Setiap pohon dilatih dengan subset berbeda dan melakukan split berdasarkan subset fitur yang dipilih secara acak pada tiap percabangan.

3. **Voting Mayoritas**
   Untuk klasifikasi, setiap pohon memberikan prediksinya, dan hasil akhir ditentukan melalui *mayoritas suara* (voting).

---

### Penjelasan Implementasi

1. **Pemrosesan Fitur dan Target**

   * Fitur yang digunakan: `ref`, `review_date`, `cocoa_percent`, `company_location`, `bean_type`, `broad_bean_origin`.
   * Target: `rating_label` (klasifikasi ke dalam kelas `low`, `medium`, dan `high`).

2. **Split Data**
   Dataset dibagi menjadi:

   * 80% data latih (`X_train`, `y_train`)
   * 20% data uji (`X_test`, `y_test`)
   * Dengan parameter `random_state=42` dan `stratify=y` untuk menjaga proporsi label.

3. **Training Model Dasar**
   Model dasar Random Forest pertama-tama dilatih menggunakan **parameter default**:

   ```python
   rf_clf = RandomForestClassifier(random_state=42)
   rf_clf.fit(X_train, y_train)
   ```

4. **Tuning Hyperparameter**
   Dilakukan pencarian parameter terbaik menggunakan `GridSearchCV` dengan *cross-validation* 3-fold dan metrik evaluasi `accuracy`. Parameter yang diuji meliputi:

   ```python
   param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [None, 10, 20],
       'min_samples_split': [2, 5],
       'min_samples_leaf': [1, 2]
   }
   ```

   Model terbaik dari hasil pencarian ini disimpan di `best_model`.

---

### Evaluasi

1. **Prediksi dan Evaluasi**
   Model terbaik (`best_model`) digunakan untuk memprediksi data uji, dan hasilnya dievaluasi menggunakan:

   * **Classification Report**
     Mencakup akurasi, precision, recall, dan f1-score untuk masing-masing kelas (`low`, `medium`, `high`).

   * **Confusion Matrix**
     Visualisasi performa model terhadap prediksi benar dan salah.

   ```python
   y_pred = best_model.predict(X_test)
   print(classification_report(y_test, y_pred))
   ```

2. **Visualisasi**
   Confusion matrix divisualisasikan menggunakan `matplotlib`:

   ```python
   cm = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['low', 'medium', 'high'])
   disp.plot(cmap='Blues')
   ```

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
