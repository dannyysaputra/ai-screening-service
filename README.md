# AI-Powered Candidate Screening Service

This project is a backend service built as part of a case study.  
Its purpose is to **automate the initial screening of job candidates** by using AI to evaluate CVs and project reports.

The service uses a **Retrieval-Augmented Generation (RAG)** architecture to compare candidate documents against *ground truth* documents (like job descriptions and scoring rubrics) to produce a structured, AI-generated evaluation report.

---

## Key Features

- **API Service (FastAPI)** – Provides RESTful endpoints for uploading documents and managing the evaluation process.  
- **AI Evaluation Pipeline** – An LLM chain that evaluates the CV and project report separately, then synthesizes the results.  
- **Asynchronous Job Queue (Celery + Redis)** – Handles long-running AI evaluations in a background worker, keeping the API responsive.  
- **RAG (Retrieval-Augmented Generation)** – Uses ChromaDB to store and retrieve "ground truth" documents (JDs, rubrics) as context for the LLM.  
- **Output Validation** – Forces LLM output into JSON format and validates it using Pydantic for stability.  

---

## Bonus Features

- **Dynamic RAG Management** – API endpoints (`GET /documents`, `POST /documents`) let you dynamically manage ground truth documents in ChromaDB.  
- **Interactive Admin Dashboard** – A simple Streamlit dashboard (`dashboard.py`) for uploading files, running evaluations, and viewing real-time results.  

---

## Design Choices & Architecture

### **Backend Framework: FastAPI**
> Chosen for its high performance, built-in async support, and automatic OpenAPI docs.

### **Job Queue: Celery + Redis**
> AI evaluations are long-running; Celery ensures persistence and reliability beyond app restarts.

### **Vector DB (RAG): ChromaDB**
> Lightweight and persistent on disk — ideal for case studies without complex infra.

### **Embedding Model: all-MiniLM-L6-v2**
> Local model via `sentence-transformers`; fast and dependency-free (no API required).

### **LLM: Google Gemini (`gemini-2.0-flash`)**
> The latest generation of Google's Gemini models, offering **higher reasoning accuracy**, **faster inference**, and full support for **JSON Mode** — allowing the AI to produce well-structured, machine-parseable output.  
> 
> This project uses `response_mime_type="application/json"` to ensure that the LLM always returns a strict JSON response, drastically improving the stability of the evaluation pipeline.


### **UI (Bonus): Streamlit**
> Enables rapid, interactive prototyping to demonstrate backend features visually.

---

## Installation & Running Instructions

Follow these steps to run the complete system (Backend API, AI Worker, and Frontend Dashboard).

### 1. Prerequisites
- Python 3.10+
- Docker (to run Redis)
- A [Google AI Studio](https://aistudio.google.com) account to get a `GEMINI_API_KEY`

---

### 2. Initial Setup

Clone this repository:
```bash
git clone https://github.com/dannyysaputra/ai-screening-service.git
cd ai-screening-service
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a .env file in the project root:
```bash
GEMINI_API_KEY=AIzaSy... (Your API Key from Google AI Studio)
```

---

### 3. Running the Services (3 Terminals)

You need 3 terminals for different services.

**Terminal 1: Run Redis (via Docker)**

This is the message broker (queue) for Celery.
```bash
docker run -d -p 6379:6379 redis
```

**Terminal 2: Run the Celery Worker**

This is the AI worker that will process the evaluation tasks.
```bash
# (Make sure your venv is active)
# IMPORTANT: If you are on Windows, add the --pool=solo flag
# to avoid a PermissionError
celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

*Wait until you see `... ready.` message.*

**Terminal 3: Run the API Server**

This is your main backend.
```bash
# (Make sure your venv is active)
uvicorn app.main:app --reload
```

*The API server runs at http://127.0.0.1:8000*

---

### 4. Ingest Documents (RAG)

Before you can evaluate, your RAG database (ChromaDB) must be populated with the ground truth documents.

**Option 1 (Recommended):** Use the dashboard (see Step 5) to upload the files from the docs_ground_truth folder one by one.

**Option 2 (Manual):** Run the ingestion script once.
```bash
# (Make sure your venv is active)
python scripts/ingest.py
```

---

### 5. (Optional) Run the Dashboard (Streamlit)

Open a 4th terminal to run the UI.
```bash
# (Make sure your venv is active)
streamlit run dashboard.py
```

*Open http://localhost:8501 in your browser to use the application.*

---

##  API Endpoint Overview

Full API documentation (Swagger) available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Main Endpoints

| **Method** | **Endpoint**      | **Description** |
|-------------|------------------|-----------------|
| **POST**    | `/upload`        | Uploads CV and project report |
| **POST**    | `/evaluate`      | Starts evaluation job (`cv_id`, `project_report_id`, `job_title`) |
| **GET**     | `/result/{id}`   | Gets job status (`queued`, `processing`, `failed`, `completed`) |

---

### Bonus Endpoints

| **Method** | **Endpoint**      | **Description** |
|-------------|------------------|-----------------|
| **GET**     | `/documents`     | Lists all ground truth docs in RAG |
| **POST**    | `/documents`     | Uploads a new ground truth doc (e.g., job description) |

## Tech Stack Summary

| **Component**        | **Technology**              |
|----------------------|-----------------------------|
| **API Framework**    | FastAPI                     |
| **Worker Queue**     | Celery + Redis              |
| **Vector Database**  | ChromaDB                    |
| **Embedding Model**  | all-MiniLM-L6-v2            |
| **LLM Model**        | Gemini 2.0 Flash            |
| **UI / Demo**        | Streamlit                   |


## Example Workflow

1. **Upload** CV + Project via **`/upload`**  
2. **Start** evaluation job via **`/evaluate`**  
3. **Check** job status via **`/result/{job_id}`**  
4. **View** structured AI feedback — CV Score, Project Score, and Final Summary


## Author

**Danny Suggi Saputra**  
*Case Study Project — AI-Powered Candidate Screening System*  

Built with ❤️ using **FastAPI**, **Celery**, **Redis**, **ChromaDB**, and **Google Gemini (2.0 Flash)**.
