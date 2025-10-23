import os
import sys
from celery import Celery

# Tambahkan path root proyek agar worker bisa impor 'app.services'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Definisikan Celery App
# broker = Antrian tugas (Redis)
# backend = Tempat menyimpan hasil (Redis)
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Impor pipeline AI 
from app.services import run_real_evaluation_pipeline

@celery_app.task(name="tasks.evaluate_candidate", bind=True)
def evaluate_candidate_task(
    self,
    cv_path: str, 
    report_path: str, 
    job_title: str
):
    """
    Wrapper Celery untuk pipeline AI Anda.
    Fungsi ini akan dijalankan oleh worker terpisah.
    """
    job_id = self.request.id
    print(f"CELERY WORKER: Menerima job {job_id}")
    print(f"Job title: {job_title}, CV path: {cv_path}, Report path: {report_path}")
    try:
        # Panggil fungsi pipeline asli
        final_result = run_real_evaluation_pipeline(
            job_id, cv_path, report_path, job_title
        )
        print(f"CELERY WORKER: Job {job_id} sukses.")
        # Kembalikan hasil. Celery akan menyimpannya di backend (Redis).
        return final_result.model_dump()
    
    except Exception as e:
        print(f"CELERY WORKER: Job {job_id} GAGAL. Error: {e}")
        # Saat exception terjadi, Celery akan menandai task sebagai 'FAILURE'
        # dan menyimpan exception sebagai hasilnya.
        raise e

# Konfigurasi opsional tapi bagus
celery_app.conf.update(
    task_track_started=True,
)