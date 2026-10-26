# Receipt Reimburse Scanner

Aplikasi lokal untuk memindai struk, meninjau hasil OCR, menyimpan konfirmasi sebagai batch, dan mengekspor laporan reimburse.

## Fitur

- Unggah dan proses beberapa foto struk sekaligus.
- Koreksi merchant, tanggal, dan total sebelum mengonfirmasi batch.
- Simpan riwayat batch secara lokal menggunakan SQLite.
- Filter riwayat berdasarkan tanggal konfirmasi dan ekspor batch terpilih ke CSV, Excel, atau PDF.

## Menjalankan aplikasi

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Database `receipt_history.db` dibuat di folder proyek saat aplikasi dijalankan. File database, foto struk contoh, hasil ekspor, dan virtual environment dikecualikan dari Git.
