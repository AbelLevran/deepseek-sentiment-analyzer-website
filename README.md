# DeepSentiment Analysis Web App 🚀

DeepSentiment adalah aplikasi web berbasis Artificial Intelligence yang dirancang secara khusus untuk melakukan analisis sentimen terhadap ulasan aplikasi DeepSeek. Aplikasi ini menggunakan model **Support Vector Machine (SVM)** yang dilatih secara *custom* untuk memprediksi apakah sebuah ulasan bernada Positif atau Negatif.

Proyek ini dibangun dengan memisahkan arsitektur *Frontend* yang sepenuhnya statis dengan *Backend* API berkinerja tinggi.

## 🤔 Kenapa Dinamakan DeepSentiment?

Nama **DeepSentiment** adalah gabungan dari dua kata:
1. **Deep (dari DeepSeek)**: Karena website ini pada awalnya dirancang khusus untuk membedah dan menganalisis ulasan pengguna aplikasi kecerdasan buatan, *DeepSeek*.
2. **Sentiment (Sentimen)**: Karena fungsi inti AI ini adalah menganalisis "sentimen" atau emosi di balik sebuah teks—apakah teks tersebut memiliki nada bicara **Positif** (pujian, puas) atau **Negatif** (kritikan, kecewa).

Singkatnya, *DeepSentiment* adalah **"Sebuah asisten pintar untuk memahami emosi dan opini publik terhadap aplikasi DeepSeek dengan sangat mendalam"**.

## 🏗️ Arsitektur & Teknologi

* **Frontend**: HTML5, CSS3 (Modern Glassmorphism Design dengan tema Hitam & Kuning khas DeepSeek), dan Vanilla JavaScript. Tidak memerlukan *library* berat seperti React, sehingga sangat ringan dan cepat.
* **Backend**: Python dengan framework **FastAPI**. Dipilih karena kecepatannya, kemudahan penggunaan, serta dokumentasi API interaktif bawaan (Swagger UI).
* **Model Machine Learning**: 
  * Linear Support Vector Classification (SVM) dari `scikit-learn`.
  * Representasi teks menggunakan `TfidfVectorizer`.
  * Prapemrosesan bahasa alami menggunakan `NLTK` (SnowballStemmer, penghapusan Stopwords) dan normalisasi kata gaul (*slang*) khusus bahasa Inggris.
  * Pelabelan data latih dilakukan menggunakan *VADER Sentiment Analyzer*.

1. **Prediksi Sentimen Tunggal (Single Predict)**: 
   * **Fungsi:** Untuk menganalisis *satu teks* atau kalimat secara instan.
   * **Kegunaan:** Jika Anda menerima ulasan yang membingungkan atau banyak menggunakan *slang* (contoh: "Aplikasi ini not bad tapi kadang ngelag"), Anda cukup menempelkannya di sini. AI akan langsung memberikan prediksi (Positif/Negatif/Netral) beserta *Confidence Score* (tingkat keyakinan) dalam hitungan detik. Ini sangat cocok untuk pengecekan cepat secara *real-time*.
2. **Prediksi Massal (Batch Predict)**: 
   * **Fungsi:** Untuk menganalisis *ratusan teks sekaligus* dalam satu kali proses (Massal).
   * **Kegunaan:** Jika Anda mendapat 100 ulasan baru dalam sehari, mengeceknya satu per satu tentu membuang waktu. Dengan fitur ini, Anda bisa mengunggah file `.csv` berisi semua ulasan tersebut. AI akan memproses semuanya sekaligus dan menyajikan tabel hasil yang rapi (bisa diunduh kembali). Ini sangat menghemat waktu riset tim Anda.
3. **Dashboard Statistik (*Static Insights*)**: 
   * **Fungsi:** Menyediakan visualisasi data menggunakan **Chart.js**, menampilkan grafik distribusi sentimen dan statistik ringkasan.
   * **Catatan Penting:** Data yang ditampilkan di dashboard ini **bersifat statis** dan diambil langsung dari dataset awal ulasan DeepSeek (file CSV bawaan yang disimpan di *backend*). Dashboard ini tidak berubah secara otomatis saat Anda melakukan *Batch Predict*, melainkan berfungsi sebagai wawasan (*insights*) keseluruhan dari data pelatihan.
4. **Riwayat Prediksi (History)**: Aplikasi menyimpan riwayat prediksi *Single Predict* Anda secara lokal di peramban web (*Local Storage*), sehingga tidak hilang jika halaman dimuat ulang.

## 🎯 Kesimpulan: Apa Kegunaan Website Ini?

Secara sederhana, **website ini adalah "Asisten Pintar Penganalisis Opini"**. 

Website ini dibuat untuk membantu **pengembang aplikasi, tim marketing, atau peneliti** dalam memahami apa yang dirasakan oleh pelanggan mereka tanpa harus membaca komentar ribuan orang satu per satu secara manual. Dengan website ini, Anda bisa tahu:
- Apakah secara umum orang-orang puas dengan produk Anda? (Dilihat melalui *Dashboard*).
- Komplain spesifik apa yang masuk hari ini? (Dianalisis secara massal melalui *Batch Predict*).
- Apakah gaya bahasa pengguna bermakna baik atau buruk? (Dianalisis dengan *Single Predict* yang didukung Normalisasi Kata Gaul).

## 📂 Struktur Direktori

```text
project-root/
│
├── backend/                  # Kode sumber API dan Model ML
│   ├── app.py                # File utama FastAPI
│   ├── requirements.txt      # Daftar dependensi Python
│   ├── model/                # Folder penyimpanan model .pkl
│   ├── preprocess/           # Skrip pembersihan teks dan kamus slang
│   ├── routes/               # Routing endpoint API (predict, dashboard)
│   └── tests/                # Unit test untuk API
│
├── frontend/                 # Kode antarmuka web (UI) statis
│   ├── index.html            # Halaman utama (Landing Page)
│   ├── css/                  # File CSS (Tema DeepSeek Glassmorphism)
│   ├── js/                   # File logika Javascript
│   └── pages/                # Halaman Single, Batch, dan Dashboard
│
├── data/                     # Folder penyimpanan dataset mentah
│
├── run.bat                   # Skrip otomatis untuk menjalankan Backend di Windows
└── train_model.py            # Skrip untuk melatih dan mengekstrak model ML
```

## 🚀 Panduan Menjalankan Aplikasi Secara Lokal

Untuk menjalankan aplikasi ini di komputer Anda, Anda perlu menyalakan Backend (API) dan membuka Frontend di browser.

### 1. Menjalankan Backend API
Pastikan Anda sudah menginstal **Python 3.9+** di komputer Anda.

1. Buka Terminal atau Command Prompt, lalu arahkan ke direktori proyek ini.
2. Instal semua pustaka Python yang dibutuhkan:
   ```cmd
   pip install -r backend/requirements.txt
   ```
3. (Opsional) Jika Anda mengubah kode pelatihan dan ingin melatih ulang model, jalankan skrip:
   ```cmd
   python train_model.py
   ```
4. **Jalankan Server Backend** dengan menjalankan file `run.bat` melalui terminal:
   ```cmd
   .\run.bat
   ```
   *Atau dengan perintah manual:*
   ```cmd
   cd backend
   python -m uvicorn app:app --reload --port 8000
   ```
   Jika berhasil, API akan berjalan di `http://localhost:8000`. Anda bisa membuka `http://localhost:8000/docs` di browser untuk melihat dan mencoba dokumentasi API interaktif.

### 2. Menjalankan Frontend
Karena frontend sepenuhnya berupa file HTML/CSS/JS statis, ada dua cara mudah untuk menjelajahinya:

**Cara Cepat:**
Buka folder `frontend` di File Explorer Anda dan klik dua kali file `index.html`. File akan terbuka langsung di browser.

**Cara Terbaik (Menggunakan Server Lokal):**
Buka terminal **baru** (biarkan terminal backend yang sebelumnya tetap menyala), lalu jalankan perintah:
```cmd
cd frontend
python -m http.server 3000
```
Buka peramban web Anda (seperti Chrome atau Edge) dan akses **http://localhost:3000**.

## 🔗 Dokumentasi API Endpoint

Berikut adalah daftar endpoint REST API yang tersedia pada Backend:

- `GET /health` : Mengecek status kesehatan server API.
- `POST /predict` : Menerima payload JSON (contoh: `{ "text": "Great app!" }`) dan mengembalikan label sentimen beserta persentase *score* keyakinan.
- `POST /predict-batch` : Menerima array teks ulasan (maksimal 100 per proses) dan mengembalikan kumpulan array hasil prediksi.
- `GET /dashboard-data` : Mengembalikan statistik ringkasan dari file CSV yang berada di folder `data/` untuk keperluan visualisasi pada halaman Dashboard frontend.


