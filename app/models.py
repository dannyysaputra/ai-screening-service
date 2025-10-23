"""
Definisi Model Pydantic
File ini berisi skema data untuk validasi input (Request)
dan output (Response) API kita.
Menggunakan Pydantic memastikan data yang bersih dan terdokumentasi
secara otomatis di Swagger.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional

# --- Model untuk Request Body ---

class EvaluateRequest(BaseModel):
    """
    [cite_start]Skema Body untuk POST /evaluate [cite: 28]
    """
    job_title: str = Field(description="Judul pekerjaan yang dilamar kandidat.", examples=["Backend Developer"])
    cv_id: str = Field(
        description="ID unik dari CV yang diupload.", 
        examples=["a1b2c3d4-..."]
    )
    project_report_id: str = Field(
        description="ID unik dari Laporan Proyek yang diupload.", 
        examples=["e5f6g7h8-..."]
    )

# --- Model untuk Response Body ---

class JobStatusQueued(BaseModel):
    """
    [cite_start]Skema Response saat job baru dibuat [cite: 30-33]
    """
    id: str = Field(description="ID unik untuk melacak job evaluasi.", examples=["job_456"])
    status: Literal["queued"] = "queued"

class JobStatusProcessing(BaseModel):
    """
    [cite_start]Skema Response untuk GET /result/{id} saat masih diproses [cite: 41-42]
    """
    id: str = Field(description="ID job yang sedang diproses.", examples=["job_456"])
    status: Literal["processing"] = "processing"

class EvaluationResultData(BaseModel):
    """
    [cite_start]Skema untuk nested object 'result' saat job selesai [cite: 47-53]
    """
    cv_match_rate: float = Field(
        description="Skor kecocokan CV (0.0 - 1.0)", 
        examples=[0.82]
    )
    cv_feedback: str = Field(
        description="Feedback kualitatif untuk CV", 
        examples=["Strong in backend..."]
    )
    project_score: float = Field(
        description="Skor Laporan Proyek (1.0 - 5.0)", 
        examples=[4.5]
    )
    project_feedback: str = Field(
        description="Feedback kualitatif untuk Laporan Proyek", 
        examples=["Meets requirements..."]
    )
    overall_summary: str = Field(
        description="Ringkasan akhir 3-5 kalimat", 
        examples=["Good candidate fit..."]
    )


class JobStatusCompleted(BaseModel):
    """
    [cite_start]Skema Response untuk GET /result/{id} saat sudah selesai [cite: 44-54]
    """
    id: str = Field(description="ID job yang sudah selesai.", examples=["job_456"])
    status: Literal["completed"] = "completed"
    result: EvaluationResultData

class JobStatusFailed(BaseModel):
    """Skema Response untuk GET /result/{id} saat job gagal"""
    id: str = Field(description="ID job yang gagal.", examples=["job_456"])
    status: Literal["failed"] = "failed"
    error: str = Field(description="Pesan error dari pipeline", examples=["Gagal mem-parsing PDF"])

class DocumentSummary(BaseModel):
    """Model Pydantic untuk daftar dokumen"""
    source_name: str
    chunk_count: int