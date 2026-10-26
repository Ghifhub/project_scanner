"""
app.py
Streamlit app: scan receipts, review results, confirm batches, and export filtered history.
Run with: streamlit run app.py
"""

import io
import sqlite3
from io import BytesIO
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.preprocess import (
    resize_max_dim, to_grayscale, denoise, deskew, adaptive_threshold, otsu_threshold,
)
from utils.ocr import extract_text_lines
from utils.parser import parse_receipt, score_total_candidate

st.set_page_config(page_title="Receipt Reimburse Scanner", layout="wide")

DATABASE_PATH = Path(__file__).with_name("receipt_history.db")


def initialize_history_db():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                confirmed_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                merchant TEXT,
                date TEXT,
                total INTEGER
            )
        """)


def save_confirmation_batch(records):
    confirmed_at = datetime.now().astimezone()
    label = f"Batch {confirmed_at.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        cursor = connection.execute(
            "INSERT INTO batches (label, confirmed_at) VALUES (?, ?)",
            (label, confirmed_at.isoformat(timespec="milliseconds")),
        )
        batch_id = cursor.lastrowid
        connection.executemany(
            """INSERT INTO receipts (batch_id, filename, merchant, date, total)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (batch_id, row["filename"], row["merchant"], row["date"], row["total"])
                for row in records
            ],
        )
        connection.commit()
        return batch_id, label
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_history_batches():
    with sqlite3.connect(DATABASE_PATH) as connection:
        return connection.execute(
            """SELECT b.id, b.label, b.confirmed_at, COUNT(r.id)
               FROM batches AS b LEFT JOIN receipts AS r ON r.batch_id = b.id
               GROUP BY b.id ORDER BY b.confirmed_at DESC, b.id DESC"""
        ).fetchall()


def get_receipts_for_batches(batch_ids):
    columns = ["Batch", "Filename", "Merchant", "Date", "Total (Rp)"]
    if not batch_ids:
        return pd.DataFrame(columns=columns)
    placeholders = ",".join("?" for _ in batch_ids)
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            f"""SELECT b.label, r.filename, r.merchant, r.date, r.total
                FROM receipts AS r JOIN batches AS b ON b.id = r.batch_id
                WHERE b.id IN ({placeholders})
                ORDER BY b.confirmed_at DESC, b.id DESC, r.id""",
            tuple(batch_ids),
        ).fetchall()
    return pd.DataFrame(rows, columns=columns)


initialize_history_db()

st.markdown(
    """
    <style>
    .top-navbar {
        position: fixed;
        top: 3.5rem;
        left: 0;
        right: 0;
        z-index: 1001;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 4rem;
        margin: 0;
        padding: 0.75rem 1.25rem;
        overflow: hidden;
        border: 1px solid rgba(147, 197, 253, 0.35);
        border-radius: 0;
        background: linear-gradient(110deg, #1d4ed8 0%, #2563eb 52%, #4f46e5 100%);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.2);
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.015em;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 8rem;
    }
    .top-navbar::before,
    .top-navbar::after {
        position: absolute;
        content: "";
        pointer-events: none;
        border-radius: 999px;
    }
    .top-navbar::before {
        width: 150px;
        height: 150px;
        top: -100px;
        left: 8%;
        background: rgba(255, 255, 255, 0.12);
    }
    .top-navbar::after {
        width: 110px;
        height: 110px;
        right: 10%;
        bottom: -82px;
        background: rgba(191, 219, 254, 0.18);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }
    </style>
    <div class="top-navbar">Receipt Reimburse Scanner</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Menu")
    page = st.radio("Navigasi", ["Scan Struk", "Riwayat"], label_visibility="collapsed")

if "records" not in st.session_state:
    st.session_state.records = []  # list of dicts, one per processed receipt


def preprocess_from_bytes(file_bytes: bytes, debug: bool = False) -> np.ndarray:
    """Same pipeline as utils.preprocess.preprocess_receipt, but from in-memory bytes
    (streamlit gives uploaded files as bytes, not a path on disk)."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File tidak bisa dibaca sebagai gambar. Coba gunakan JPG atau PNG.")
    img = resize_max_dim(img)
    gray = to_grayscale(img)
    gray = denoise(gray)
    gray = deskew(gray)
    return gray if debug else adaptive_threshold(gray)


def preprocess_variants_from_bytes(file_bytes: bytes, use_binarized: bool) -> dict[str, np.ndarray]:
    """Prepare two OCR views so one thresholding method cannot hide receipt text."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File tidak bisa dibaca sebagai gambar. Coba gunakan JPG atau PNG.")
    gray = deskew(denoise(to_grayscale(resize_max_dim(img))))
    if use_binarized:
        return {
            "adaptive": adaptive_threshold(gray),
            "otsu": otsu_threshold(gray),
        }
    return {
        "grayscale": gray,
        "otsu": otsu_threshold(gray),
    }


def create_excel_report(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Laporan Reimburse")
        workbook = writer.book
        worksheet = writer.sheets["Laporan Reimburse"]
        header_format = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#2563EB",
            "border": 0, "align": "center", "valign": "vcenter",
        })
        money_format = workbook.add_format({"num_format": '"Rp" #,##0'})
        for col_num, column in enumerate(df.columns):
            worksheet.write(0, col_num, column, header_format)
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
        column_widths = {
            "Batch": 28,
            "Filename": 24,
            "Merchant": 28,
            "Date": 16,
            "Total (Rp)": 20,
        }
        for col_num, column in enumerate(df.columns):
            cell_format = money_format if column == "Total (Rp)" else None
            worksheet.set_column(col_num, col_num, column_widths.get(column, 18), cell_format)
        worksheet.set_default_row(20)
        worksheet.set_row(0, 24)
    return output.getvalue()


def create_pdf_report(df: pd.DataFrame, total: int) -> bytes:
    output = BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4),
        rightMargin=14 * mm, leftMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Laporan Reimburse", styles["Title"]),
        Paragraph(f"Jumlah struk: {len(df)} &nbsp; | &nbsp; Total: Rp {total:,.0f}".replace(",", "."), styles["Normal"]),
        Spacer(1, 8 * mm),
    ]

    def pdf_text(value):
        text = "" if pd.isna(value) else str(value)
        return text.encode("latin-1", "replace").decode("latin-1")

    rows = [[pdf_text(column) for column in df.columns]]
    for _, row in df.iterrows():
        rows.append([
            "" if column == "Total (Rp)" and pd.isna(row[column])
            else f"Rp {int(row[column]):,}".replace(",", ".") if column == "Total (Rp)"
            else pdf_text(row[column])
            for column in df.columns
        ])
    widths_mm = {"Batch": 48, "Filename": 48, "Merchant": 82, "Date": 32, "Total (Rp)": 38}
    table = Table(
        rows,
        colWidths=[widths_mm.get(column, 40) * mm for column in df.columns],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF6FF")]),
        ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    doc.build(story)
    return output.getvalue()


def render_export_buttons(df: pd.DataFrame, total: int):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_bytes = create_excel_report(df)
    pdf_bytes = create_pdf_report(df, total)
    csv_col, excel_col, pdf_col = st.columns(3)
    with csv_col:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="reimburse_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with excel_col:
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name="reimburse_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with pdf_col:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="reimburse_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# Clear old review widget state before widgets are recreated for a later scan.
for widget_key in st.session_state.pop("review_widget_keys_to_clear", []):
    if widget_key in st.session_state:
        del st.session_state[widget_key]

if page == "Scan Struk":
    uploaded_files = st.file_uploader(
        "Upload struk (bisa lebih dari satu)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    use_binarized = st.checkbox(
        "Bandingkan OCR binarized (adaptive + Otsu; hasil dipilih otomatis)",
        value=False,
    )

    if uploaded_files and st.button("Proses semua struk"):
        # Streamlit keeps widget values by key across reruns. Use a fresh key
        # namespace for every scan so old review values cannot mask new OCR data.
        previous_scan_id = st.session_state.get("scan_id", 0)
        for idx in range(len(st.session_state.get("records", []))):
            for field in ("merchant", "date", "total"):
                st.session_state.pop(f"{field}_{previous_scan_id}_{idx}", None)
        st.session_state.scan_id = previous_scan_id + 1
        st.session_state.records = []
        progress = st.progress(0.0)
        failures = []
        for i, uf in enumerate(uploaded_files):
            file_bytes = uf.getvalue()
            try:
                variant_results = []
                for mode, processed in preprocess_variants_from_bytes(file_bytes, use_binarized).items():
                    lines = extract_text_lines(processed)
                    parsed = parse_receipt(lines)
                    variant_results.append({
                        "mode": mode,
                        "lines": lines,
                        "parsed": parsed,
                        "total_score": score_total_candidate(lines, parsed.total),
                    })
                if not variant_results:
                    raise ValueError("Gambar tidak menghasilkan teks OCR.")

                # Choose totals from the OCR pass with the clearest textual
                # evidence; choose the date from any pass that recognized one.
                best_total_result = max(
                    variant_results,
                    key=lambda result: (result["total_score"], result["parsed"].total or 0),
                )
                best_date_result = next(
                    (result for result in sorted(
                        variant_results,
                        key=lambda item: item["total_score"],
                        reverse=True,
                    ) if result["parsed"].date),
                    best_total_result,
                )
                parsed = best_total_result["parsed"]
                raw_text = "\n\n".join(
                    f"[{result['mode']}]\n" + "\n".join(result["lines"])
                    for result in variant_results
                )
                st.session_state.records.append({
                    "filename": uf.name,
                    "merchant": parsed.merchant,
                    "date": best_date_result["parsed"].date,
                    "total": parsed.total,
                    "raw_text": raw_text,
                    "preview_bytes": file_bytes,
                })
            except Exception as exc:
                failures.append(f"{uf.name}: {exc}")
            progress.progress((i + 1) / len(uploaded_files))
        if failures:
            st.error("Sebagian file gagal diproses:\n\n" + "\n".join(f"- {error}" for error in failures))
        succeeded = len(uploaded_files) - len(failures)
        if succeeded:
            st.success(f"Selesai memproses {succeeded} dari {len(uploaded_files)} struk.")

    if st.session_state.records:
        st.subheader("Review & koreksi hasil")

        for idx, rec in enumerate(st.session_state.records):
            with st.expander(f"Struk: {rec['filename']}", expanded=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(Image.open(io.BytesIO(rec["preview_bytes"])), use_container_width=True)

                with col2:
                    scan_id = st.session_state.get("scan_id", 0)
                    rec["merchant"] = st.text_input("Merchant", rec["merchant"] or "", key=f"merchant_{scan_id}_{idx}")
                    rec["date"] = st.text_input("Tanggal (YYYY-MM-DD)", rec["date"] or "", key=f"date_{scan_id}_{idx}")
                    total_val = st.text_input("Total (Rp)", str(rec["total"] or ""), key=f"total_{scan_id}_{idx}")
                    try:
                        rec["total"] = int(total_val) if total_val else None
                    except ValueError:
                        st.warning("Total harus angka.")

                    with st.expander("Lihat teks OCR mentah"):
                        st.text(rec["raw_text"])

        st.divider()
        st.caption("Konfirmasi menyimpan semua hasil review sebagai satu batch di riwayat lokal.")
        if st.button("Konfirmasi & simpan batch", type="primary"):
            try:
                batch_id, batch_label = save_confirmation_batch(st.session_state.records)
                st.session_state.review_widget_keys_to_clear = [
                    f"{field}_{st.session_state.get('scan_id', 0)}_{idx}"
                    for idx in range(len(st.session_state.records))
                    for field in ("merchant", "date", "total")
                ]
                st.session_state.records = []
                st.success(f"{batch_label} berhasil disimpan ke riwayat.")
            except Exception as exc:
                st.error(f"Batch gagal disimpan; tidak ada data yang dikonfirmasi. Detail: {exc}")
    else:
        st.info("Upload minimal satu foto struk lalu klik 'Proses semua struk'.")

else:
    st.subheader("Riwayat konfirmasi")
    batches = get_history_batches()
    if not batches:
        st.info("Belum ada batch tersimpan. Proses dan konfirmasi struk dari halaman Scan Struk.")
    else:
        batch_rows = []
        for batch_id, label, confirmed_at, count in batches:
            timestamp = datetime.fromisoformat(confirmed_at).astimezone()
            batch_rows.append({
                "id": batch_id,
                "label": label,
                "timestamp": timestamp,
                "tanggal": timestamp.date(),
                "Jumlah struk": count,
            })

        min_date = min(row["tanggal"] for row in batch_rows)
        max_date = max(row["tanggal"] for row in batch_rows)
        date_range = st.date_input(
            "Filter tanggal konfirmasi",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="history_timestamp_range",
        )

        if isinstance(date_range, tuple):
            if len(date_range) == 2:
                start_date = date_range[0] or min_date
                end_date = date_range[1] or max_date
            elif len(date_range) == 1:
                start_date = end_date = date_range[0] or min_date
            else:
                start_date, end_date = min_date, max_date
        else:
            start_date = end_date = date_range or min_date
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        visible_batches = [
            row for row in batch_rows
            if start_date <= row["tanggal"] <= end_date
        ]

        if not visible_batches:
            st.info("Tidak ada batch pada rentang tanggal yang dipilih.")
        else:
            if "selected_history_batches" not in st.session_state:
                st.session_state.selected_history_batches = []
            selected_batches = st.session_state.selected_history_batches

            header_cols = st.columns([2.4, 2, 0.9, 1.6])
            for column, heading in zip(header_cols, ("Batch", "Waktu konfirmasi", "Struk", "Aksi")):
                column.markdown(f"**{heading}**")

            for row in visible_batches:
                is_selected = row["id"] in selected_batches
                with st.container(border=True):
                    batch_col, time_col, count_col, action_col = st.columns([2.4, 2, 0.9, 1.6])
                    batch_col.write(row["label"])
                    time_col.write(row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
                    count_col.write(row["Jumlah struk"])
                    if action_col.button(
                        "Hapus dari laporan" if is_selected else "Tambah ke laporan",
                        key=f"toggle_batch_{row['id']}",
                        use_container_width=True,
                    ):
                        if is_selected:
                            selected_batches.remove(row["id"])
                        else:
                            selected_batches.append(row["id"])
                        st.session_state.selected_history_batches = selected_batches
                        st.rerun()

            st.caption(f"{len(selected_batches)} batch dipilih untuk laporan.")

            if not selected_batches:
                st.info("Tambahkan minimal satu batch melalui tombol pada tabel untuk melihat dan mengekspor laporannya.")
            else:
                history_df = get_receipts_for_batches(selected_batches)
                if history_df.empty:
                    st.info("Batch yang dipilih tidak memiliki data struk.")
                else:
                    st.subheader("Struk dalam batch terpilih")
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                    total_reimburse = int(history_df["Total (Rp)"].fillna(0).sum())
                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Jumlah struk", len(history_df))
                    metric_col2.metric("Total reimburse", f"Rp {total_reimburse:,.0f}".replace(",", "."))
                    st.subheader("Ekspor batch terpilih")
                    render_export_buttons(history_df, total_reimburse)
