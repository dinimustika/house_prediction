# Laporan Proyek Machine Learning - Dini Mustika
## Project Overview
Menentukan harga sewa atau harga jual rumah merupakan satu kegiatan yang memerlukan konsentrasi dan menguras tenaga. Dengan banyaknya faktor penentu yang ada, seperti luas tanah dan bangunan, banyak kamar dan fasilitas lainnya yang akan menentukan harga tersebut. Dengan banyaknya faktor-faktor penentu harga, masyarakat awam tentu akan kesulitan untuk menentukan harga apabila ingin menjual atau menyewakan rumah mereka. Berangkat dari masalah ini, maka akan lebih mudah apabila ada suatu alat untuk dapat memprediksi harga sewa atau harga jual rumah.

## Business Understanding

### Problem Statements

<!-- Menjelaskan pernyataan masalah latar belakang: -->
- Bagaimana cara menentukan harga jual atau harga sewa rumah?
- Berapa harga kisaran untuk rumah dengan jumlah kamar tidur tertentu?
- Fasilitas apa yang berpengaruh pada harga jual atau harga sewa rumah?

### Goals

<!-- Menjelaskan tujuan dari pernyataan masalah: -->
- Mengetahui cara menentukan harga jual dan/atau harga sewa rumah
- Mengetahui kisaran harga untuk rumah dengan jumlah kamar tidur tertentu
- Mengetahui fasilitas apa yang berpengaruh pada harga jual dan/atau harga sewa rumah

<!-- Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan. -->

<!-- **Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:  -->

### Solution statements
- Menggunakan algoritma K-Nearest Neighbors dan Random Forest untuk memprediksi harga dan/atau harga sewa rumah  

## Data Understanding
<!-- Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data). -->
Pada proyek kali ini, dataset yang digunakan adalah House Rent dataset yang didapatkan dari [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

<!-- Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:   -->

### Variabel-variabel pada House Rent dataset adalah sebagai berikut:
- BHK: Jumlah Kamar Tidur, Aula, Dapur
- Rent: Sewa Rumah/Apartemen/Rumah Susun
- Size: Ukuran Rumah/Apartemen/Rumah Susun dalam Meter Persegi
- Floor: Rumah/Apartemen/Rumah Susun yang terletak di lantai mana dan Jumlah Lantai (Contoh: Ground dari 2, 3 dari 5, dll.)
- Area Type: Ukuran Rumah/Apartemen/Rumah Susun dihitung pada Super Area atau Carpet Area atau Build Area.
- Area Locality: Lokalitas Rumah/Apartemen/Rumah Susun
- City: Kota dimana Rumah/Apartemen/Rumah Susun berada
- Furnishing Status: atus Perabotan Rumah/Apartemen/Rumah Susun, Furnished atau Semi-Furnished atau Unfurnished.
- Tenant Preferred: Jenis Tenant yang lebih disukai oleh Pemilik
- Bathroom: Jumlah Kamar Mandi
- Point of Contact: Siapa yang harus Anda hubungi untuk informasi lebih lanjut mengenai Rumah/Apartemen/Rumah Susun
<!-- **Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis. -->

### Exploratory Data Analysis
Untuk melakukan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data, saya menggunakan EDA. 
- Keadaan deskriptif dari dataset <br>
Dibawah ini menunjukkan keadaan deskriptif dari data, meliputi mean, min, max dsb
![image](https://user-images.githubusercontent.com/73211764/190621090-b3378823-0ca9-475b-a702-76b2d19ea677.png)

- Missing values <br>
Dibawah ini menunjukkan apakah terdapat missing value didalam dataset <br>
![image](https://user-images.githubusercontent.com/73211764/190622409-46928d02-1365-4bb9-8d38-b523401ffd23.png)

- Korelasi antara numerical features dalam dataset <br>
Korelasi dalam dataset ini ditunjukkan melalui histogram <br>
![image](https://user-images.githubusercontent.com/73211764/190622846-f306a37c-507d-4b9c-bd74-da827a00b470.png)

- Menangani Outliers <br>
![image](https://user-images.githubusercontent.com/73211764/190628691-1e2ce304-21ef-46b5-9228-8a5e910afd68.png) <br>
![image](https://user-images.githubusercontent.com/73211764/190628757-f993b8fc-dbf0-4885-a549-484be1ae6c91.png) <br>
Menangani outliers yang ada dengan menggunakan metode IQR <br>
![image](https://user-images.githubusercontent.com/73211764/190628846-2eb5d3ea-ecdb-4255-9635-01f59508a82c.png)

- Categorical Features <br>
Dibawah ini menunjukkan pengaruh harga sewa terhadap masing-masing fitur <br>
![image](https://user-images.githubusercontent.com/73211764/190623374-a0824351-9486-40ed-b218-167c501bff19.png)
![image](https://user-images.githubusercontent.com/73211764/190623420-cf59a52b-a025-46ea-90d0-3c8d798af9bb.png)
![image](https://user-images.githubusercontent.com/73211764/190623503-c6fcab3f-f435-4c28-adad-7a3e12390904.png)
![image](https://user-images.githubusercontent.com/73211764/190623580-24214e82-0ae9-4c67-917c-0529c6ec0dd9.png)


## Data Preparation
<!-- Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan. -->

1. Encoding fitur kategori <br>
Bagian ini akan membuat variabel kategori kita telah berubah menjadi variabel numerik
![image](https://user-images.githubusercontent.com/73211764/190624021-7cf65c81-3d23-4a26-b8d4-d32e1bea377b.png)

3. Train test split <br>
![image](https://user-images.githubusercontent.com/73211764/190624081-4df89469-0d3b-4a1a-befe-9950965268dd.png)

5. Standardisasi <br>
Proses scaling dan standarisasi berguna untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma <br>
![image](https://user-images.githubusercontent.com/73211764/190624119-24fc3133-6d56-40d1-8e83-f6f5499024c5.png)

Bentuk deskriptif data setelah distandarisasi <br>
![image](https://user-images.githubusercontent.com/73211764/190624502-06aa5415-d87c-4200-8dad-3f20aa2946b6.png)


<!-- **Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut. -->

## Modeling

<!-- Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan. -->

1. K-Nearest Neighbors <br>
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. <br>
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). <br>
![image](https://user-images.githubusercontent.com/73211764/190625751-bce17cb3-9caa-4ee4-9742-75d935e0b9f5.png) <br>

Kelebihan dari KNN adalah: <br>
- Sederhana untuk dipahami dan diterapkan
- Model yang terus berkembang: Saat diekspos ke data baru, model berubah untuk mengakomodasi titik data baru.
- Masalah multi-kelas juga dapat diselesaikan.
- Satu Parameter Hyper: K-NN mungkin memerlukan waktu saat memilih hyper parameter

Kekurangan:
- Lambat untuk kumpulan data besar.
- Curse of dimensionality: Tidak bekerja dengan baik pada kumpulan data dengan banyak fitur.
- Penskalaan data mutlak harus.
- Tidak bekerja dengan baik pada data yang tidak seimbang. Jadi sebelum menggunakan k-NN baik kelas undersamplemajority maupun kelas minoritas oversample dan memiliki dataset yang seimbang.
- Peka terhadap outlier.
- Tidak dapat menangani missing values dengan baik

2. Random Forest <br>
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Pada proyek kali ini, teknik ensemble yang digunakan adalah bootstrap <br>
![image](https://user-images.githubusercontent.com/73211764/190625987-7b1c03c6-ace7-4abe-9359-abbb087c6894.png) <br>
  
Kelebihan dari Random Forest adalah: <br>
  - Random forest dapat melakukan decorrelate trees. 
  - Mengurangi kesalahan: Random forest adalah kumpulan pohon keputusan. Untuk memprediksi hasil dari baris tertentu, random forest mengambil input dari semua trees dan kemudian memprediksi hasilnya
  - Performa yang baik pada dataset yang tidak seimbang : Ini juga dapat menangani kesalahan dalam data yang tidak seimbang 
  - Penanganan data dalam jumlah besar: Dapat menangani data dalam jumlah besar dengan dimensi variabel yang lebih tinggi.
  - Penanganan data yang hilang dengan baik: Dapat menangani data yang hilang dengan sangat baik. Jadi jika ada banyak data yang hilang dalam model, itu akan memberikan hasil yang baik.
  - Dampak kecil dari outlier: Karena hasil akhir diambil dengan berkonsultasi dengan banyak decision trees, titik data tertentu yang merupakan outlier tidak akan berdampak besar pada Random Forest.
  - Tidak ada masalah overfitting: Dalam Random forest hanya mempertimbangkan subset fitur, dan hasil akhir bergantung pada semua trees. Jadi ada lebih banyak generalisasi dan lebih sedikit overfitting.
  - Berguna untuk mengekstrak fitur penting

  Kekurangan: <br>
  - Fitur harus memiliki kekuatan prediktif jika tidak, fitur tersebut tidak akan berfungsi.
  - Prediksi trees harus tidak berkorelasi.
  - Muncul sebagai Black Box: Sulit untuk mengetahui apa yang terjadi.




<!-- **Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**. -->

## Evaluation
Metrik yang akan saya gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. <br>
![image](https://user-images.githubusercontent.com/73211764/190628136-59fa90c9-8700-420e-836e-053ec73a3fa7.png) <br>
Tampilan menggunakan bar chart <br>
![image](https://user-images.githubusercontent.com/73211764/190628288-26b9c083-7ab1-4976-b420-f003ebde962e.png)

Hasil prediksi diantara dua model: <br>
![image](https://user-images.githubusercontent.com/73211764/190628431-9c6fb481-047f-4f74-9053-292e1f4578a8.png)

<!-- Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan. -->