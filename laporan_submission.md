# Laporan Proyek Machine Learning - Dini Mustika
## Project Overview
Menentukan harga sewa atau harga jual rumah merupakan satu kegiatan yang memerlukan konsentrasi dan menguras tenaga. Dengan banyaknya faktor penentu yang ada, seperti luas tanah dan bangunan, banyak kamar dan fasilitas lainnya yang akan menentukan harga tersebut. Dengan banyaknya faktor-faktor penentu harga, masyarakat awam tentu akan kesulitan untuk menentukan harga apabila ingin menjual atau menyewakan rumah mereka. Berangkat dari masalah ini, maka akan lebih mudah apabila ada suatu alat untuk dapat memprediksi harga sewa atau harga jual rumah.

## Business Understanding

### Problem Statements

- Bagaimana cara menentukan harga jual atau harga sewa rumah?
- Berapa harga kisaran untuk rumah dengan jumlah kamar tidur tertentu?
- Fasilitas apa yang berpengaruh pada harga jual atau harga sewa rumah?

### Goals

- Mengetahui cara menentukan harga jual dan/atau harga sewa rumah
- Mengetahui kisaran harga untuk rumah dengan jumlah kamar tidur tertentu
- Mengetahui fasilitas apa yang berpengaruh pada harga jual dan/atau harga sewa rumah

### Solution statements
- Menggunakan algoritma *K-Nearest Neighbors* dan *Random Forest* untuk memprediksi harga dan/atau harga sewa rumah  

## Data Understanding
Pada proyek kali ini, dataset yang digunakan adalah House Rent dataset yang didapatkan dari [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset). Dataset ini merupakan kumpulan data rumah/properti dari berbagai kota yang ada di India. Memiliki sekitar 4000+ baris dan 11 kolom.

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

### Exploratory Data Analysis
Untuk melakukan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data, saya menggunakan EDA. 
- Mengetahui keadaan deskriptif dari dataset <br>
Untuk menunjukkan keadaan deskriptif dari data, saya menggunakan fungsi describe() yang dapat mengembalikan nilai meliputi nilai *count*, mean, *standard deviation*, quantil data dan lain sebagainya <br>
![image](https://user-images.githubusercontent.com/73211764/190621090-b3378823-0ca9-475b-a702-76b2d19ea677.png)

- Menangani *Missing values* <br>
Untuk menunjukkan apakah terdapat missing value didalam dataset, saya menggunakan fungsi bawaan yang dimiliki oleh *pandas* library. Pada bagian ini akan mengembalikan nilai True ataupun False. False memiliki arti tidak terdapat missing values dan True memiliki arti sebaliknya <br>
![image](https://user-images.githubusercontent.com/73211764/190622409-46928d02-1365-4bb9-8d38-b523401ffd23.png)

- Mengetahui korelasi antara numerical features dalam dataset <br>
Korelasi dalam dataset ini ditunjukkan melalui heatmap. Dengan heatmap ini, dapat terlihat hubungan antara fitur numerik yang ada <br>
![image](https://user-images.githubusercontent.com/73211764/190622846-f306a37c-507d-4b9c-bd74-da827a00b470.png)

- Menangani *Outliers* <br>
![image](https://user-images.githubusercontent.com/73211764/190628691-1e2ce304-21ef-46b5-9228-8a5e910afd68.png) <br>
![image](https://user-images.githubusercontent.com/73211764/190628757-f993b8fc-dbf0-4885-a549-484be1ae6c91.png) <br>
Dari diagram tersebut, terdapat beberapa nilai yang jaraknya jauh dari nilai rata-rata sehingga diindikasikan terdapat outliers pada data tersebut. Pada proyek kali ini, untuk menangani outliers yang ada adalah dengan menggunakan metode IQR. <br>
Metode IQR(*Interquartile Range*) adalah sebuah metode yang membagi data menjadi kuartil-kuartil yang sama besar. Misalnya quartil pertama (Q1), kedua (Q2) atau lebih dikenal dengan Median, dan quantil ketiga (Q3). Setelah metode IQR dilakukan, maka bentuk dari dataframe akan berubah, ditunjukkan dengan gambar di bawah ini <br>
![image](https://user-images.githubusercontent.com/73211764/190837070-8f7f20cb-350c-4a67-8fd5-1adb050e4ed5.png)

<!-- ![image](https://user-images.githubusercontent.com/73211764/190628846-2eb5d3ea-ecdb-4255-9635-01f59508a82c.png) -->

- Categorical Features <br>
Dibawah ini menunjukkan pengaruh harga sewa terhadap masing-masing fitur <br>
![image](https://user-images.githubusercontent.com/73211764/190623374-a0824351-9486-40ed-b218-167c501bff19.png)
![image](https://user-images.githubusercontent.com/73211764/190623420-cf59a52b-a025-46ea-90d0-3c8d798af9bb.png)
![image](https://user-images.githubusercontent.com/73211764/190623503-c6fcab3f-f435-4c28-adad-7a3e12390904.png)
![image](https://user-images.githubusercontent.com/73211764/190623580-24214e82-0ae9-4c67-917c-0529c6ec0dd9.png)


## Data Preparation

1. Encoding fitur kategori <br>
Pada bagian ini, saya menggunakan One Hot Encoder yang telah disediakan oleh library scikit-learn dan hasil akhir pada bagian ini adalah membuat variabel kategori kita telah berubah menjadi variabel numerik
![image](https://user-images.githubusercontent.com/73211764/190624021-7cf65c81-3d23-4a26-b8d4-d32e1bea377b.png)

2. Train test split <br>
Pada bagian ini, saya menggunakan library *scikit-learn* untuk membagi dataset menjadi *Training Set* dan *Testing Set*. Pada proyek ini, saya membagi testing set sebesar 10% dan training set merupakan sisa data yang ada yaitu 90%
<!-- ![image](https://user-images.githubusercontent.com/73211764/190624081-4df89469-0d3b-4a1a-befe-9950965268dd.png) -->

3. Standardisasi <br>
Proses *scaling* dan standarisasi berguna untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Metode yang saya gunakan adalah StandardScaler, yang mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.
<!-- ![image](https://user-images.githubusercontent.com/73211764/190624119-24fc3133-6d56-40d1-8e83-f6f5499024c5.png) -->

Setelah distandarisasi, maka bentuk deskriptif data menjadi seperti di bawah ini <br>
![image](https://user-images.githubusercontent.com/73211764/190624502-06aa5415-d87c-4200-8dad-3f20aa2946b6.png)


## Modeling

1. ### K-Nearest Neighbors <br>

KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. <br>
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Pada proyek ini, saya menggunakan sebanyak 10 k tetangga terdekat dengan metric Euclidean untuk mengukur jarak antar titik. <br>
<!-- ![image](https://user-images.githubusercontent.com/73211764/190625751-bce17cb3-9caa-4ee4-9742-75d935e0b9f5.png) <br> -->

Kelebihan dari KNN adalah: <br>
- Sederhana untuk dipahami dan diterapkan
- Model yang terus berkembang: Saat diekspos ke data baru, model berubah untuk mengakomodasi titik data baru.
- Masalah multi-kelas juga dapat diselesaikan.
- Satu Parameter Hyper: K-NN mungkin memerlukan waktu saat memilih hyper parameter

Kekurangan:
- Lambat untuk kumpulan data besar.
- *Curse of dimensionality*: Tidak bekerja dengan baik pada kumpulan data dengan banyak fitur.
- Penskalaan data mutlak harus.
- Tidak bekerja dengan baik pada data yang tidak seimbang. Jadi sebelum menggunakan k-NN baik kelas undersamplemajority maupun kelas minoritas oversample dan memiliki dataset yang seimbang.
- Peka terhadap *outlier*.
- Tidak dapat menangani missing values dengan baik

2. ### *Random Forest* <br>
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble *(group) learning*. Dalam *random forest* teknik pendekatan ensemble dikenal dengan *bagging* dan *boosting*. Pada proyek kali ini saya menggunakan *bagging*. Pada model ini, saya menggunakan berbagai parameter dengan nilai sebagai berikut:
- n_estimators=50, merujuk kepada seberapa banyak *trees* yang akan digunakan
- max_depth=16, merujuk kepada level maksimum setiap *trees*
- random_state=55
- n_jobs=-1<br>
<!-- ![image](https://user-images.githubusercontent.com/73211764/190625987-7b1c03c6-ace7-4abe-9359-abbb087c6894.png) <br> -->

  
Kelebihan dari *Random Forest* adalah: <br>
  - Random forest dapat melakukan *decorrelate trees*. 
  - Mengurangi kesalahan: Random forest adalah kumpulan pohon keputusan. Untuk memprediksi hasil dari baris tertentu, random forest mengambil input dari semua *trees* dan kemudian memprediksi hasilnya
  - Performa yang baik pada dataset yang tidak seimbang : Ini juga dapat menangani kesalahan dalam data yang tidak seimbang 
  - Penanganan data dalam jumlah besar: Dapat menangani data dalam jumlah besar dengan dimensi variabel yang lebih tinggi.
  - Penanganan data yang hilang dengan baik: Dapat menangani data yang hilang dengan sangat baik. Jadi jika ada banyak data yang hilang dalam model, itu akan memberikan hasil yang baik.
  - Dampak kecil dari outlier: Karena hasil akhir diambil dengan berkonsultasi dengan banyak decision *trees*, titik data tertentu yang merupakan outlier tidak akan berdampak besar pada *Random Forest*.
  - Tidak ada masalah overfitting: Dalam Random forest hanya mempertimbangkan subset fitur, dan hasil akhir bergantung pada semua *trees*. Jadi ada lebih banyak generalisasi dan lebih sedikit *overfitting*.
  - Berguna untuk mengekstrak fitur penting

  Kekurangan: <br>
  - Fitur harus memiliki kekuatan prediktif jika tidak, fitur tersebut tidak akan berfungsi.
  - Prediksi *trees* harus tidak berkorelasi.
  - Muncul sebagai *Black Box*: Sulit untuk mengetahui apa yang terjadi.


## Evaluation
Metrik yang akan saya gunakan pada prediksi ini adalah MSE atau *Mean Squared Error* yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Semakin tinggi nilai yang didapat maka semakin buruk model tersebut. Sebagai perbandingan, nilai MSE dari RF lebih kecil jika dibandingkan dengan KNN<br>
![image](https://user-images.githubusercontent.com/73211764/190628136-59fa90c9-8700-420e-836e-053ec73a3fa7.png) <br>
Tampilan menggunakan bar chart <br>
![image](https://user-images.githubusercontent.com/73211764/190628288-26b9c083-7ab1-4976-b420-f003ebde962e.png)

Hasil prediksi diantara dua model: <br>
![image](https://user-images.githubusercontent.com/73211764/190628431-9c6fb481-047f-4f74-9053-292e1f4578a8.png)

Dengan ini, dapat dikatakan bahwa algoritma *Random Forest* mendapatkan hasil prediksi yang lebih baik dibandingkan dengan algoritma KNN

Referensi: <br>
[1]  Sergey. arayev, et al. (2021). "Full Stack Deep Learning". [Online]. Diakses pada 17 September 2022. <br>
[2] Rogati, Monica. "The AI Hierarchy of Needs". Diakses pada 17 September 2022. <br>
[3] Gabor Mellis, et al. “On the State of the Art of Evaluation in Neural Language Models”. 2018. ICLR. <br>
[4] Kelleher, John D, et al. "Machine Learning for Predictive Data Analytics". MIT Press. 2020. <br>
[5] IBM Cloud. "Exploratory Data Analysis". Diakses pada 17 September 2022. <br>
[6] Kang, Hyun. "The Prevention and Handling the Missing Data". Diakses pada 17 September 2022. <br>
[7] Kuhn, Max dan Johnson Kjell. "Applied Predictive Modeling". Springer. 2013. <br>
[8] Gupta, Shailaja. "Pros and cons of various Machine Learning algorithms". Diakses 17 September 2022 <br>
