# Singapore Air Quality Dashboard

Dashboard pemantauan kualitas udara real-time di Singapura, dibangun dengan Streamlit, Apache Airflow, dan PostgreSQL. Data bersumber dari OpenAQ v3 dan Open-Meteo Historical Archive.

---

## Arsitektur

Pipeline data mengikuti pola Medallion Architecture:

- **Bronze** — data mentah dari API (OpenAQ + Open-Meteo), disimpan apa adanya
- **Silver** — data yang sudah dibersihkan dan distandarisasi per jam per lokasi
- **Gold** — view analitik akhir yang digunakan oleh dashboard

Orkestrasi pipeline dijalankan oleh Apache Airflow. Streamlit membaca langsung dari PostgreSQL.

---

## Struktur Proyek

```
case-method-2/
├── app/
│   └── streamlit_app.py        # Dashboard Streamlit
├── dags/
│   └── env_pipeline_dag.py     # Airflow DAG
├── sql/
│   └── init_db.sql             # Inisialisasi skema database
├── .env                        # Variabel lingkungan
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Prasyarat

- Docker dan Docker Compose
- API key dari [OpenAQ](https://openaq.org/)

---

## Cara Menjalankan

```bash
cp .env.example .env
```

```
POSTGRES_DB=envdata
POSTGRES_USER=envuser
POSTGRES_PASSWORD=your_password

POSTGRES_HOST=postgres
POSTGRES_PORT=5432

OPENAQ_API_KEY=your_openaq_api_key

START_DATE=start_date
END_DATE=end_date
```

**2. Jalankan semua layanan**

```bash
docker compose up --build -d
```

**3. Akses antarmuka**

| Layanan   | URL                     | Keterangan                    |
|-----------|-------------------------|-------------------------------|
| Dashboard | http://localhost:8501   | Streamlit                     |
| Airflow   | http://localhost:8080   | Username & password: `admin`  |

**4. Jalankan pipeline data**

Buka Airflow di browser, aktifkan DAG `env_data_pipeline`, lalu trigger secara manual. Pipeline akan mengambil data dari OpenAQ dan Open-Meteo sesuai rentang tanggal di `.env`.

---

## Fitur Dashboard

**Tab Dashboard**

- Metrik ringkasan: rata-rata PM2.5, suhu, kelembaban, jumlah rekaman
- Grafik time series PM2.5 per stasiun
- Peta persebaran stasiun (PyDeck) dengan warna berdasarkan tingkat keparahan
- Bar chart konsentrasi rata-rata PM2.5 per stasiun
- Heatmap pola diurnal PM2.5
- Scatter plot suhu vs PM2.5 dan kelembaban vs PM2.5
- Tabel ringkasan stasiun dan akses data mentah

**Tab ML Insights**

- Prakiraan PM2.5 12 jam ke depan menggunakan Ridge Regression dengan fitur lag
- Deteksi anomali menggunakan IsolationForest
- Fitur importance dan matriks korelasi menggunakan RandomForest

---

## Stasiun Pemantauan

| Nama Stasiun             | Lokasi (lat, lon)         |
|--------------------------|---------------------------|
| 461B AQ                  | 1.3555, 103.7403          |
| Midwood                  | 1.3641, 103.7637          |
| NASA GSFC Rutgers Calib. | 1.2976, 103.7803          |
| Shelford                 | 1.3250, 103.8125          |
| Singapore                | 1.3521, 103.8198          |
| Potong Pasir             | 1.3309, 103.8687          |
| Joo Chiat Place          | 1.3137, 103.9018          |
| Ocean Park               | 1.3095, 103.9179          |

---

## Sumber Data

- [OpenAQ v3](https://openaq.org/) — data kualitas udara (PM1, PM2.5, PM10)
- [Open-Meteo Historical Archive](https://open-meteo.com/) — data cuaca (suhu, kelembaban)

---

## Teknologi

| Komponen    | Teknologi                                  |
|-------------|---------------------------------------------|
| Dashboard   | Streamlit, Plotly, PyDeck                   |
| Pipeline    | Apache Airflow                              |
| Database    | PostgreSQL 15                               |
| ML          | scikit-learn (Ridge, IsolationForest, RF)   |
| Container   | Docker, Docker Compose                      |
