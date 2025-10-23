import os
import re
import json
import time
import sys
from typing import Dict, List, Any, Type, TypeVar 
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi import HTTPException
import chromadb
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.generative_models import GenerativeModel
from pydantic import BaseModel, ValidationError
import io
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Konfigurasi Awal ---

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.models import EvaluationResultData
load_dotenv()

# --- Konstanta ---
CHROMA_DB_PATH = os.path.join(project_root, "chroma_db")
COLLECTION_NAME = "screening_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
LLM_MODEL_NAME = "gemini-2.0-flash"

# Ini akan memperbaiki semua error 'BaseModel' vs 'CvEvaluationOutput'
T = TypeVar("T", bound=BaseModel)

# --- Pydantic Models (Tidak Berubah) ---

class CvEvaluationOutput(BaseModel):
    cv_match_rate: float
    cv_feedback: str

class ProjectEvaluationOutput(BaseModel):
    project_score: float
    project_feedback: str

class FinalSummaryOutput(BaseModel):
    overall_summary: str

# --- Klien Servis (Chroma & Gemini) ---

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embedding_func = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func # type: ignore
    )
    print(f"Koneksi ke ChromaDB collection '{COLLECTION_NAME}' berhasil.")
except Exception as e:
    print(f"CRITICAL: Gagal terhubung ke ChromaDB. Error: {e}")
    collection = None 

try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env")
    
    genai.configure(api_key=GEMINI_API_KEY) # type: ignore
    
    generation_config = GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json"
    )
    llm_model = GenerativeModel(
        model_name=LLM_MODEL_NAME,
        generation_config=generation_config
    )
    print(f"Koneksi ke Gemini model '{LLM_MODEL_NAME}' berhasil.")
except Exception as e:
    print(f"CRITICAL: Gagal mengkonfigurasi Gemini. Error: {e}")
    llm_model = None

# --- Fungsi Helper ---

def parse_pdf(file_path: str) -> str:
    """Membaca file PDF dan mengembalikan isinya sebagai satu string."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        print(f"Berhasil mem-parsing {file_path}")
        return text
    except Exception as e:
        print(f"Gagal mem-parsing PDF di {file_path}: {e}")
        raise

def query_rag(query_text: str, sources: List[str], n_results: int = 5) -> str:
    """Membuat kueri ke ChromaDB untuk konteks RAG."""
    if collection is None:
        raise Exception("Koneksi ChromaDB tidak tersedia.")
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"source": {"$in": sources}} # type: ignore
        )
        
        if not results or not isinstance(results, dict):
            print("RAG query: results kosong atau bukan dict")
            return ""

        docs = results.get("documents")
        if not docs or not docs[0]:
            print("RAG query: tidak ada dokumen relevan")
            return ""

        context = "\n---\n".join(docs[0])
        return context

    except Exception as e:
        print(f"Gagal RAG query: {e}")
        raise

def call_gemini_with_retry(prompt: str, retries: int = 3, delay: int = 5) -> str:
    """Memanggil API LLM dengan retry logic sederhana."""
    if llm_model is None:
        raise Exception("Model LLM (Gemini) tidak terinisialisasi.")
    
    for attempt in range(retries):
        try:
            print(f"Memanggil LLM (Attempt {attempt + 1}/{retries})...")
            response = llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gagal memanggil LLM: {e}")
            if attempt < retries - 1:
                print(f"Mencoba lagi dalam {delay} detik...")
                time.sleep(delay)
            else:
                print("Gagal memanggil LLM setelah beberapa kali percobaan.")
                raise 
    return ""  # NOTE: Pylance mungkin masih error di sini, tapi ini unreachable.

def parse_llm_json(
    llm_output: str, 
    output_model: Type[T]
) -> T:
    """
    Mem-parsing output JSON dari LLM dan memvalidasinya dengan model Pydantic.
    """
    json_data: Dict[str, Any] = {}
    try:
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not match:
            print(f"Peringatan: Output LLM bukan JSON, mencoba parsing langsung: {llm_output[:50]}...")
            json_str = llm_output
        else:
            json_str = match.group(0)

        json_data = json.loads(json_str)
        
        # Pylance sekarang tahu 'validated_data' adalah Tipe 'T' (misal: CvEvaluationOutput)
        validated_data = output_model(**json_data)
        print(f"Berhasil mem-parsing dan validasi output LLM untuk {output_model.__name__}")
        return validated_data
        
    except json.JSONDecodeError as e:
        print(f"FATAL: Gagal mem-parsing JSON dari LLM. Error: {e}. Output mentah: {llm_output}")
        raise Exception(f"Invalid JSON response from LLM: {e}")
    except ValidationError as e:
        print(f"FATAL: Output JSON LLM tidak sesuai skema. Error: {e}. Data: {json_data}")
        raise Exception(f"LLM output schema mismatch: {e}")
    except Exception as e:
        print(f"FATAL: Error tidak diketahui saat parsing. Error: {e}")
        raise

# --- Desain Prompt ---

def create_cv_prompt(cv_text: str, jd_context: str, rubric_context: str) -> str:
    """Membuat prompt untuk evaluasi CV"""
    output_schema = CvEvaluationOutput.model_json_schema()
    
    return f"""
    Anda adalah seorang Manajer Perekrutan Teknis senior.
    Tugas Anda adalah mengevaluasi CV kandidat berdasarkan Deskripsi Pekerjaan (Job Description) dan Rubrik Penilaian CV.
    
    Hitung skor rata-rata tertimbang (skala 1-5) berdasarkan Rubrik, lalu KONVERSIKAN ke skala 0.0 - 1.0 (Skor / 5.0).
    Berikan feedback yang jujur dan ringkas.

    KONTEKS PENTING:
    --- Rubrik Penilaian CV ---
    {rubric_context}
    
    --- Deskripsi Pekerjaan (Job Description) ---
    {jd_context}

    DATA KANDIDAT:
    --- CV Kandidat (Teks) ---
    {cv_text}
    
    INSTRUKSI OUTPUT:
    Anda HARUS memberikan jawaban HANYA dalam format JSON yang valid, sesuai dengan skema ini.
    JANGAN tambahkan salam, penjelasan, atau teks lain di luar blok JSON.
    
    Skema JSON:
    {json.dumps(output_schema, indent=2)}
    """

def create_project_prompt(report_text: str, brief_context: str, rubric_context: str) -> str:
    """Membuat prompt untuk evaluasi Laporan Proyek"""
    output_schema = ProjectEvaluationOutput.model_json_schema()
    
    return f"""
    Anda adalah seorang Senior Backend Engineer.
    Tugas Anda adalah mengevaluasi Laporan Proyek kandidat berdasarkan Case Study Brief (soal) dan Rubrik Penilaian Proyek.
    
    Hitung skor rata-rata tertimbang (skala 1.0 - 5.0) berdasarkan Rubrik.
    Berikan feedback yang jujur dan ringkas.

    KONTEKS PENTING:
    --- Rubrik Penilaian Proyek ---
    {rubric_context}
    
    --- Case Study Brief (Soal) ---
    {brief_context}

    DATA KANDIDAT:
    --- Laporan Proyek Kandidat (Teks) ---
    {report_text}
    
    INSTRUKSI OUTPUT:
    Anda HARUS memberikan jawaban HANYA dalam format JSON yang valid, sesuai dengan skema ini.
    JANGAN tambahkan salam, penjelasan, atau teks lain di luar blok JSON.
    
    Skema JSON:
    {json.dumps(output_schema, indent=2)}
    """

def create_summary_prompt(cv_eval: CvEvaluationOutput, project_eval: ProjectEvaluationOutput) -> str:
    """Membuat prompt untuk analisis akhir/ringkasan"""
    output_schema = FinalSummaryOutput.model_json_schema()
    
    return f"""
    Anda adalah seorang Hiring Manager.
    Anda telah menerima dua laporan evaluasi untuk seorang kandidat.
    Tugas Anda adalah mensintesis kedua evaluasi ini menjadi satu ringkasan akhir (3-5 kalimat)
    yang menyoroti kekuatan, kelemahan, dan rekomendasi.
    
    EVALUASI 1: CV
    {cv_eval.model_dump_json(indent=2)}
    
    EVALUASI 2: Laporan Proyek
    {project_eval.model_dump_json(indent=2)}
    
    INSTRUKSI OUTPUT:
    Anda HARUS memberikan jawaban HANYA dalam format JSON yang valid, sesuai dengan skema ini.
    JANGAN tambahkan salam, penjelasan, atau teks lain di luar blok JSON.
    
    Skema JSON:
    {json.dumps(output_schema, indent=2)}
    """


# --- FUNGSI UTAMA PIPELINE ---

def run_real_evaluation_pipeline(
    job_id: str, 
    cv_path: str, 
    report_path: str, 
    job_title: str,
):
    """
    Orkestrator pipeline AI yang sesungguhnya.
    """
    print(f"MEMULAI PIPELINE NYATA untuk job: {job_id}...")
    
    try:
        print(f"Job {job_id} status: processing")
        
        cv_text = parse_pdf(cv_path)
        report_text = parse_pdf(report_path)
        
        print(f"Job {job_id}: Memulai Evaluasi CV...")
        jd_context = query_rag(
            query_text=f"skills untuk {job_title}", 
            sources=["job_description"]
        )
        cv_rubric_context = query_rag(
            query_text="rubrik penilaian cv", 
            sources=["cv_rubric"]
        )
        cv_prompt = create_cv_prompt(cv_text, jd_context, cv_rubric_context)
        cv_response_str = call_gemini_with_retry(cv_prompt)
        cv_eval = parse_llm_json(cv_response_str, CvEvaluationOutput)
        
        print(f"Job {job_id}: Memulai Evaluasi Proyek...")
        brief_context = query_rag(
            query_text="persyaratan case study", 
            sources=["case_study_brief"]
        )
        project_rubric_context = query_rag(
            query_text="rubrik penilaian proyek", 
            sources=["project_rubric"]
        )
        project_prompt = create_project_prompt(
            report_text, brief_context, project_rubric_context
        )
        project_response_str = call_gemini_with_retry(project_prompt)
        project_eval = parse_llm_json(project_response_str, ProjectEvaluationOutput)
        
        print(f"Job {job_id}: Memulai Analisis Akhir...")
        summary_prompt = create_summary_prompt(cv_eval, project_eval)
        summary_response_str = call_gemini_with_retry(summary_prompt)
        summary_eval = parse_llm_json(summary_response_str, FinalSummaryOutput)
        
        final_result = EvaluationResultData(
            cv_match_rate=cv_eval.cv_match_rate,
            cv_feedback=cv_eval.cv_feedback,
            project_score=project_eval.project_score,
            project_feedback=project_eval.project_feedback,
            overall_summary=summary_eval.overall_summary
        )
        
        print(f"PIPELINE SUKSES untuk job: {job_id}.")
        return final_result

    except Exception as e:
        print(f"PIPELINE GAGAL untuk job: {job_id}. Error: {e}")
        raise e

def _get_chroma_collection():
    """Helper untuk mendapatkan koneksi collection."""
    if collection: # 'collection' dari global scope
        return collection
        
    # Jika 'collection' gagal di-load saat startup, coba lagi
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_func = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func # type: ignore
        )
    except Exception as e:
        print(f"Gagal mendapatkan Chroma collection: {e}")
        return None

def list_ground_truth_documents() -> List[Dict[str, Any]]:
    """Mengambil daftar dokumen 'source' unik dari ChromaDB."""
    coll = _get_chroma_collection()
    if not coll:
        return []
    
    metadatas = coll.get(include=["metadatas"])['metadatas']
    if not metadatas:
        return []
        
    # Buat daftar 'source' unik dan jumlah chunk-nya
    sources = {}
    for meta in metadatas:
        source_name = meta.get('source', 'unknown')
        if source_name not in sources:
            sources[source_name] = 0
        sources[source_name] += 1
    
    return [{"source_name": name, "chunk_count": count} for name, count in sources.items()]

def ingest_document(
    file_bytes: bytes, 
    file_name: str, 
    source_name: str
) -> Dict[str, Any]:
    """Memproses dan memasukkan (ingest) dokumen baru ke ChromaDB."""
    
    content = ""
    try:
        if file_name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes)) # Baca dari bytes
            for page in reader.pages:
                content += page.extract_text() + "\n\n"
        elif file_name.endswith(".txt"):
            content = file_bytes.decode('utf-8')
        else:
            raise ValueError("Tipe file tidak didukung. Hanya .pdf atau .txt")
    except Exception as e:
        print(f"Gagal mem-parsing file {file_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Gagal mem-parsing file: {e}")

    # Logika splitting
    from scripts.ingest import split_text_into_chunks, generate_document_id
    
    chunks = split_text_into_chunks(content, source_name)
    
    all_chunks_text = []
    all_metadatas = []
    all_ids = []
    
    for chunk in chunks:
        chunk_text = chunk['text']
        chunk_id = generate_document_id(chunk_text)
        all_chunks_text.append(chunk_text)
        all_metadatas.append(chunk['metadata'])
        all_ids.append(chunk_id)

    # Upsert ke ChromaDB
    coll = _get_chroma_collection()
    if not coll:
        raise HTTPException(status_code=500, detail="Database Chroma tidak tersedia")
        
    coll.upsert(
        documents=all_chunks_text,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    return {
        "message": "Dokumen berhasil di-ingest",
        "source_name": source_name,
        "chunks_added": len(all_chunks_text)
    }