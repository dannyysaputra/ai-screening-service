import os
import uuid
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Path, Form
)
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Union, List

# --- Impor Model ---
from app.models import (
    EvaluateRequest, 
    JobStatusQueued, 
    JobStatusProcessing, 
    JobStatusCompleted,
    JobStatusFailed,
    DocumentSummary,
    EvaluationResultData
)

# --- Impor Celery ---
from app.worker import celery_app, evaluate_candidate_task
from celery.result import AsyncResult # Untuk cek status

# --- Impor Servis Bonus ---
from app.services import (
    list_ground_truth_documents, 
    ingest_document
)

# --- Inisialisasi Aplikasi ---
app = FastAPI(
    title="AI Screening Service",
    description="Layanan backend untuk mengevaluasi CV dan Laporan Proyek kandidat secara otomatis.",
    version="2.0.0 (Bonus Version)"
)

# --- Konfigurasi ---
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Database in-memory Hanya untuk melacak file.
file_database: Dict[str, Dict] = {}

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Endpoint root untuk cek status server."""
    return {"message": "AI Screening Service is running."}


@app.post("/upload", 
          status_code=201, 
          tags=["1. Upload"],
          summary="Upload CV dan Laporan Proyek")
async def upload_files(
    cv: UploadFile = File(..., description="Candidate's CV in PDF format"),
    project_report: UploadFile = File(..., description="Candidate's Project Report in PDF format")
):
    """
    Menerima upload CV dan Project Report (PDF).
    Menyimpan file dan mengembalikan ID unik untuk setiap file.
    """
    # (Logika tidak berubah... ini sudah benar)
    if cv.content_type != "application/pdf" or project_report.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is allowed.")
    try:
        cv_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())
        cv_filepath = os.path.join(UPLOADS_DIR, f"{cv_id}_cv.pdf")
        project_filepath = os.path.join(UPLOADS_DIR, f"{project_id}_project.pdf")

        with open(cv_filepath, "wb") as buffer:
            buffer.write(await cv.read())
        with open(project_filepath, "wb") as buffer:
            buffer.write(await project_report.read())

        file_database[cv_id] = {"filename": f"{cv_id}_cv.pdf", "path": cv_filepath}
        file_database[project_id] = {"filename": f"{project_id}_project.pdf", "path": project_filepath}

        return JSONResponse(status_code=201, content={
            "message": "Files uploaded successfully",
            "cv_id": cv_id,
            "project_report_id": project_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")


@app.post("/evaluate", 
          response_model=JobStatusQueued, 
          tags=["2. Evaluate"],
          summary="Memulai proses evaluasi AI (via Celery)")
async def evaluate_candidate(
    request: EvaluateRequest
):
    """
    Memicu pipeline evaluasi AI secara asynchronous via Celery.
    Segera mengembalikan Task ID Celery sebagai `job_id`.
    """
    if request.cv_id not in file_database:
        raise HTTPException(status_code=404, detail=f"CV with ID {request.cv_id} not found.")
    if request.project_report_id not in file_database:
        raise HTTPException(status_code=404, detail=f"Project Report with ID {request.project_report_id} not found.")

    cv_path = file_database[request.cv_id]['path']
    report_path = file_database[request.project_report_id]['path']

    # Panggil Celery task dengan .delay()
    print(f"Mendaftarkan task ke Celery...")
    task = evaluate_candidate_task.delay( #type: ignore
        cv_path, 
        report_path, 
        request.job_title
    )
    
    # kembalikan ID dari Celery.
    # 'task.id' adalah ID unik dari Celery, misal: 'a8a3-...'
    print(f"Task {task.id} telah dimasukkan ke antrian (queued).")
    return JobStatusQueued(id=task.id, status="queued")


# Tipe Union untuk Swagger agar tahu semua kemungkinan response
EvaluationResponse = Union[JobStatusCompleted, JobStatusProcessing, JobStatusQueued, JobStatusFailed]

@app.get("/result/{id}", 
         response_model=EvaluationResponse,
         tags=["3. Result"],
         summary="Mengambil status dan hasil evaluasi (dari Celery)")
async def get_evaluation_result(
    id: str = Path(..., description="ID job yang didapat dari /evaluate", example="a8a3...")
):
    """
    Mengambil status dan hasil dari job evaluasi berdasarkan ID Task Celery.
    """
    # Cek status task di backend Celery (Redis)
    print(f"Mengecek status untuk task_id: {id}")
    task_result = AsyncResult(id, app=celery_app)
    
    status = task_result.status

    if status == "SUCCESS":
        result_data = task_result.get() 
        
        if not result_data or not isinstance(result_data, dict):
            # Error internal jika task sukses tapi tidak mengembalikan dict
            raise HTTPException(
                status_code=500,
                detail=f"Job {id} status 'SUCCESS' but result data is invalid or missing."
            )
            
        # Pydantic akan memvalidasi dict ini
        return JobStatusCompleted(
            id=id,
            status="completed",
            result=EvaluationResultData(**result_data)
        )
    elif status == "FAILURE":
        # Task gagal, ambil error-nya
        error_message = str(task_result.info) # .info berisi exception
        return JobStatusFailed(id=id, status="failed", error=error_message)
    elif status == "STARTED" or status == "RETRY":
        # Task sedang berjalan
        return JobStatusProcessing(id=id, status="processing")
    elif status == "PENDING":
        # Task ada di antrian tapi belum dimulai
        return JobStatusQueued(id=id, status="queued")
    
    # Menangani kasus jika ID tidak ditemukan oleh Celery
    raise HTTPException(status_code=404, detail=f"Job with ID {id} not found.")


@app.get("/documents", 
         response_model=List[DocumentSummary],
         tags=["4. Bonus: Documents (RAG)"],
         summary="Melihat daftar semua Dokumen Ground Truth")
async def get_documents():
    """
    Mengambil daftar semua 'source' dokumen (JD, Rubrik)
    yang saat ini ada di Vector DB (Chroma).
    """
    return list_ground_truth_documents()

@app.post("/documents", 
          tags=["4. Bonus: Documents (RAG)"],
          summary="Upload Dokumen Ground Truth baru")
async def upload_document(
    file: UploadFile = File(..., description="File .pdf atau .txt"),
    source_name: str = Form(..., description="Nama unik untuk dokumen ini (misal: 'jd_data_scientist')")
):
    """
    Meng-upload Job Description atau Rubrik baru ke Vector DB (Chroma).
    Ini akan di-chunk dan di-embed secara otomatis.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nama file tidak ditemukan")
        
    file_bytes = await file.read()
    
    try:
        result = ingest_document(file_bytes, file.filename, source_name)
        return JSONResponse(status_code=201, content=result)
    except Exception as e:
        # Menangkap error dari service (misal: file tidak didukung)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Gagal ingest dokumen: {str(e)}")


# Perintah untuk menjalankan server
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
