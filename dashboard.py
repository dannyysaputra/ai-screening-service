import streamlit as st
import requests
import time
import os

# Konfigurasi URL API
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(layout="wide")
st.title("🤖 Dashboard AI Screening Karyawan")

# --- Bagian 1: Evaluasi ---
st.header("1. Evaluasi Kandidat Baru")

col1, col2 = st.columns(2)
with col1:
    cv_file = st.file_uploader("Upload CV (PDF)", type="pdf", key="cv")
    job_title = st.text_input("Judul Pekerjaan", "Product Engineer (Backend)")
    
with col2:
    report_file = st.file_uploader("Upload Laporan Proyek (PDF)", type="pdf", key="report")
    
if st.button("Mulai Evaluasi", use_container_width=True):
    if cv_file and report_file and job_title:
        with st.spinner("1/4 - Mengupload file..."):
            try:
                files = {
                    'cv': (cv_file.name, cv_file, 'application/pdf'),
                    'project_report': (report_file.name, report_file, 'application/pdf')
                }
                upload_res = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=10)
                upload_res.raise_for_status()
                upload_data = upload_res.json()
                st.session_state['cv_id'] = upload_data['cv_id']
                st.session_state['project_report_id'] = upload_data['project_report_id']
                st.success(f"✅ File terupload: CV ID: {upload_data['cv_id']}")
            except Exception as e:
                st.error(f"Gagal upload: {e}")
                st.stop()

        with st.spinner("2/4 - Memulai job evaluasi..."):
            try:
                eval_payload = {
                    'job_title': job_title,
                    'cv_id': st.session_state['cv_id'],
                    'project_report_id': st.session_state['project_report_id']
                }
                eval_res = requests.post(f"{API_BASE_URL}/evaluate", json=eval_payload, timeout=10)
                eval_res.raise_for_status()
                eval_data = eval_res.json()
                st.session_state['job_id'] = eval_data['id']
                st.success(f"✅ Job dimulai! Job ID: {eval_data['id']}")
            except Exception as e:
                st.error(f"Gagal memulai evaluasi: {e}")
                st.stop()
        
        # --- Polling (Mengecek hasil) ---
        st.header("2. Hasil Evaluasi")
        job_id = st.session_state['job_id']
        result_placeholder = st.empty()
        
        with st.spinner(f"3/4 - Menunggu hasil untuk Job ID: {job_id}... (Ini bisa memakan waktu 1-2 menit)"):
            status = ""
            while status not in ["completed", "failed"]:
                try:
                    time.sleep(5) # Jeda 5 detik
                    result_res = requests.get(f"{API_BASE_URL}/result/{job_id}", timeout=10)
                    result_data = result_res.json()
                    status = result_data.get("status")
                    
                    # Tampilkan status mentah
                    result_placeholder.json(result_data)
                    
                    if status == "completed":
                        st.success("Evaluasi Selesai!")
                        st.balloons()
                    elif status == "failed":
                        st.error(f"Evaluasi Gagal: {result_data.get('error')}")
                
                except Exception as e:
                    st.warning(f"Gagal mengambil status: {e}")
                    time.sleep(10) # Jika error, tunggu lebih lama

    else:
        st.warning("Harap upload CV, Laporan Proyek, dan isi Judul Pekerjaan.")

# --- Bagian 2: Manajemen Dokumen ---
st.divider()
st.header("Manajemen Dokumen Ground Truth (RAG)")

col3, col4 = st.columns([1, 2])

with col3:
    st.subheader("Upload Dokumen Baru")
    doc_file = st.file_uploader("Upload .pdf atau .txt", type=["pdf", "txt"], key="doc")
    doc_source_name = st.text_input("Nama Sumber (misal: 'jd_frontend_2025')", key="doc_name")
    
    if st.button("Upload ke Vector DB", use_container_width=True):
        if doc_file and doc_source_name:
            with st.spinner(f"Meng-ingest {doc_source_name}..."):
                try:
                    doc_files = {'file': (doc_file.name, doc_file, doc_file.type)}
                    doc_data = {'source_name': doc_source_name}
                    doc_res = requests.post(
                        f"{API_BASE_URL}/documents", 
                        files=doc_files, 
                        data=doc_data,
                        timeout=30
                    )
                    doc_res.raise_for_status()
                    st.success(f"Berhasil: {doc_res.json().get('message')}")
                    st.rerun() # Refresh halaman untuk update daftar
                except Exception as e:
                    st.error(f"Gagal ingest: {e}")
        else:
            st.warning("Harap pilih file dan beri nama sumber.")

with col4:
    st.subheader("Dokumen di Vector DB")
    if st.button("Refresh Daftar Dokumen"):
        st.cache_data.clear()
        
    @st.cache_data(ttl=60) # Cache selama 60 detik
    def get_doc_list():
        try:
            res = requests.get(f"{API_BASE_URL}/documents", timeout=10)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            st.error(f"Gagal mengambil daftar dokumen: {e}")
            return []
            
    docs = get_doc_list()
    st.dataframe(docs, use_container_width=True)