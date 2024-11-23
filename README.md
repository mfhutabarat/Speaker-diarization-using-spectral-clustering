# Penerapan Pre-trained Model pada Speaker Diarization untuk Menentukan Jumlah Pembicara dalam Rekaman Rapat menggunakan Algoritma Spectral Clustering

Penerapan ini dilakukan menggunakan pendekatan pre-trained model pada speaker diarization dengan menggunakan memori yang sedikit dapat digunakan untuk menentukan jumlah pembicara berdasarkan hasil klasterisasi optimal.

## Metode

### 1. Pengumpulan Data
Data pada penelitian ini merupakan rekaman rapat yang didapatkan dari laman YouTube dengan nama akun [@GitLab Unfiltered](https://www.youtube.com/@GitLabUnfiltered). Data terdiri dari dua file
audio rekaman rapat yang berdurasi secara berurutan yaitu [33 menit 27 detik](https://www.youtube.com/watch?v=JWxQNiLYqIQ) dan [22 menit 38 detik](https://www.youtube.com/watch?v=WYMgl3JLJ4E&t=9s).

### 2. Implementasi Model
Penelitian ini menggunakan dua jenis model pre-trained open-source yaitu [Faster Whisper](https://github.com/SYSTRAN/faster-whisper/tree/master)
(re-implementasi model Whisper OpenAI) dan Speaker [Embedding](https://huggingface.co/pyannote/embedding) dari Pyannote.audio.

### 3. Reduksi Data
Data embedding setiap teks yang dihasilkan memiliki fitur sebanyak 512 fitur sehingga perlu dilakukan reduksi fitur menjadi lebih sedikit, tapi tanpa harus
menghilangkan fitur-fitur penting pada data embedding. Selanjutnya, embedding yang telah digabungkan tersebut direduksi menjadi data dengan dua dimensi
menggunakan algoritma *[T-distributed Stochastic Neighbor Embedding (TSNE)](https://scikit-learn.org/0.16/modules/generated/sklearn.manifold.TSNE.html)*.

### 4. Klasterisasi
Klasterisasi dilakukan dengan tujuan untuk menentukan banyaknya pembicara berbeda pada audio rekaman rapat. Klasterisasi data embedding yang
dihasilkan pada proses speaker embedding sebelumnya dilakukan dengan menggunakan algoritma [*Spectral Clustering*](https://scikit-learn.org/0.15/modules/generated/sklearn.cluster.SpectralClustering.html).

### 5. Evaluasi Klasterisasi
Metode Davies Bouldin Index (DBI) digunakan untuk melakukan evaluasi kualitas klaster pada data embedding suara pembicara dal rekaman rapat. 
Proses evaluasi dilakukan dengan menggunakan [*davies_bouldin_score*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) dari Scikit-Learn.

## Hasil
### 1. Data Rekaman 
Rekaman pertama <br />
<img src="https://github.com/mfhutabarat/Speaker-diarization-using-spectral-clustering/blob/main/results/rekaman%201.png" width=465  height=287 >
<br />
Rekaman kedua <br />
<img src="https://github.com/mfhutabarat/Speaker-diarization-using-spectral-clustering/blob/main/results/rekaman%202.png" width=465 height=287>

### 2. Implementasi Model
1. Faster Whisper <br />
Hasil transkripsi dan segmentasi  dari kedua rekaman <br />
<img src="https://github.com/mfhutabarat/Speaker-diarization-using-spectral-clustering/blob/main/results/hasil%20transkripsi%20faster%20whisper.png">

2. Embedding Pyannote <br />
<img src="https://github.com/mfhutabarat/Speaker-diarization-using-spectral-clustering/blob/main/results/hasil%20embedding.png" >
   
### 3. Reduksi Embedding
<img src="https://github.com/mfhutabarat/Speaker-diarization-using-spectral-clustering/blob/main/results/hasil%20reduksi%20embedding.png" >

### 4. Klasterisasi
Hasil klasterisasi dengan nilai k-klaster = 11 <br />
<img src="https://github.com/mfhutabarat/Speaker-diarization-using-spectral-clustering/blob/main/results/visualisasi%20klaster%2011.png" >

### 5. Hasil Evaluasi Davies Bouldin Index
Tabel berikut menunjukkan hasil evaluasi klasterisasi yang dilakukan pada embedding audio tidak berulang. Embedding yang direduksi menggunakan
perplexity sama dengan 15 dan n_klaster sama dengan 11 merupakan klaster dengan nilai DBI terendah dengan nilai 0.3794792717.

| No | Perplexity | Klaster | davies_bouldin_values |
|----|------------|---------|-----------------------|
| 1 | 5 | 4 | 2.964164048 |
| 2 | 5 | 5 | 3.991425591 |
| 3 | 5 | 6 | 4.365893415 |
| 4 | 5 | 7 | 1.611147753 |
| 5 | 5 | 8 | 2.80636251 |
| 6 | 5 | 9 | 1.901439554 |
| 7 | 5 | 10 | 3.624329999 |
| 8 | 5 | 11 | 1.010581134 |
| 9 | 5 | 12 | 0.9900377999 |
| 10 | 5 | 13 | 1.230448772 |
| 11 | 10 | 4 | 1.895338813 |
| 12 | 10 | 5 | 1.42589054 |
| 13 | 10 | 6 | 0.614928438 |
| 14 | 10 | 7 | 0.5430062686 |
| 15 | 10 | 8 | 1.055878336 |
| 16 | 10 | 9 | 0.4487844301 |
| 17 | 10 | 10 | 0.5331145259 |
| 18 | 10 | 11 | 0.4056098601 |
| 19 | 10 | 12 | 0.4050269432 |
| 20 | 10 | 13 | 0.4154136037 |
| 21 | 15 | 4 | 0.8364729122 |
| 22 | 15 | 5 | 1.150809286 |
| 23 | 15 | 6 | 0.5716240059 |
| 24 | 15 | 7 | 0.6029306322 |
| 25 | 15 | 8 | 0.5293891361 |
| 26 | 15 | 9 | 0.448447908 |
| 27 | 15 | 10 | 0.3836821827 |
| **28** | **15** | **11** | **0.3794792717** |
| 29 | 15 | 12 | 0.4506196754 |
| 30 | 15 | 13 | 0.5053909196 |
| 31 | 20 | 4 | 0.8693226185 |
| 32 | 20 | 5 | 0.6420146921 |
| 33 | 20 | 6 | 0.5671076136 |
| 34 | 20 | 7 | 0.6251801712 |
| 35 | 20 | 8 | 0.5268953036 |
| 36 | 20 | 9 | 0.4620421366 |
| 37 | 20 | 10 | 0.3928224404 |
| 38 | 20 | 11 | 0.3915598916 |
| 39 | 20 | 12 | 0.5615102305 |
| 40 | 20 | 13 | 0.5000367271 |
| 41 | 25 | 4 | 1.010840588 |
| 42 | 25 | 5 | 0.8858678943 |
| 43 | 25 | 6 | 0.5621953496 |
| 44 | 25 | 7 | 0.6046438694 |
| 45 | 25 | 8 | 0.5214470189 |
| 46 | 25 | 9 | 0.4535618303 |
| 47 | 25 | 10 | 0.395294526 |
| 48 | 25 | 11 | 0.3966120986 |
| 49 | 25 | 12 | 0.5285274808 |
| 50 | 25 | 13 | 0.503207634 |
| 51 | 30 | 4 | 0.4446832304 |
| 52 | 30 | 5 | 0.5104979336 |
| 53 | 30 | 6 | 0.5555601879 |
| 54 | 30 | 7 | 0.5684378517 |
| 55 | 30 | 8 | 0.4848515805 |
| 56 | 30 | 9 | 0.4523594871 |
| 57 | 30 | 10 | 0.4113646549 |
| 58 | 30 | 11 | 0.4119548244 |
| 59 | 30 | 12 | 0.4595837411 |
| 60 | 30 | 13 | 0.522887883 |
