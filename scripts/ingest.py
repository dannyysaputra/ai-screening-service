import os
import sys
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv

# --- Konfigurasi ---
load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Konstanta ---
GROUND_TRUTH_DIR = os.path.join(project_root, "docs_ground_truth")
CHROMA_DB_PATH = os.path.join(project_root, "chroma_db")
COLLECTION_NAME = "screening_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

# --- Setup Klien & Model ---

def get_embedding_function():
    """
    Membuat embedding function untuk ChromaDB menggunakan SentenceTransformer lokal.
    Model akan diunduh secara otomatis oleh library saat pertama kali dijalankan.
    """
    print(f"Menggunakan model embedding lokal: {EMBEDDING_MODEL_NAME}")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

def init_chroma_client():
    """
    Inisialisasi klien ChromaDB yang persistent (disimpan ke disk).
    """
    print(f"Inisialisasi ChromaDB di: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

# --- Logika Pemuatan & Pemecahan Dokumen ---

def load_documents_from_directory(directory_path):
    print(f"Memuat dokumen dari: {directory_path}")
    documents = []
    
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        content = ""
        source_name = os.path.splitext(filename)[0] 

        try:
            if filename.endswith(".txt"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"  - Sukses memuat {filename}")
                    
            elif filename.endswith(".pdf"):
                reader = PdfReader(filepath)
                for page in reader.pages:
                    content += page.extract_text() + "\n\n" 
                print(f"  - Sukses memuat {filename} ({len(reader.pages)} halaman)")
                
            if content:
                documents.append({"source": source_name, "content": content})
                
        except Exception as e:
            print(f"  - Gagal memuat {filename}: {e}")
            
    return documents

def split_text_into_chunks(text, source_name, chunk_size=1000, chunk_overlap=100):
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {"source": source_name}
                })
            current_chunk = paragraph + "\n\n"
            
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": {"source": source_name}
        })
        
    print(f"    - Memecah '{source_name}' menjadi {len(chunks)} chunk")
    return chunks

def generate_document_id(text_chunk):
    return hashlib.md5(text_chunk.encode('utf-8')).hexdigest()

# --- Fungsi Utama (Orchestrator) ---

def main():
    """
    Fungsi utama untuk menjalankan pipeline ingesti:
    1. Inisialisasi ChromaDB & Embedding Function (lokal)
    2. Dapatkan atau buat 'collection'
    3. Muat dokumen dari disk
    4. Pecah dokumen menjadi chunks
    5. Tambahkan chunks ke ChromaDB
    """
    print("--- Memulai Proses Ingesti RAG (Mode Stabil/Lokal) ---")
    
    # Inisialisasi Klien
    chroma_client = init_chroma_client()
    embedding_func = get_embedding_function()
    
    # Dapatkan atau Buat Collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func  # type: ignore
    )
    print(f"Collection '{COLLECTION_NAME}' siap digunakan.")

    raw_documents = load_documents_from_directory(GROUND_TRUTH_DIR)
    
    # Siapkan data untuk ChromaDB
    all_chunks_text = []
    all_metadatas = []
    all_ids = []
    
    print("\nMemecah dokumen menjadi chunks...")
    for doc in raw_documents:
        chunks = split_text_into_chunks(doc['content'], doc['source'])
        
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_id = generate_document_id(chunk_text)
            
            all_chunks_text.append(chunk_text)
            all_metadatas.append(chunk['metadata'])
            all_ids.append(chunk_id)

    if not all_chunks_text:
        print("Tidak ada dokumen yang ditemukan untuk di-ingest. Selesai.")
        return

    # Tambahkan Chunks ke ChromaDB
    print(f"\nMenambahkan {len(all_chunks_text)} chunk ke ChromaDB (mungkin perlu mengunduh model)...")
    try:
        collection.upsert(
            documents=all_chunks_text,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print("--- Proses Ingesti Selesai Sukses ---")
        print(f"Total dokumen di collection: {collection.count()}")
        
    except Exception as e:
        print(f"Error saat menambahkan dokumen ke ChromaDB: {e}")
        print("\n*** GAGAL. Kemungkinan karena koneksi internet terblokir saat mengunduh model.")
        print("*** COBA LAGI menggunakan koneksi internet lain (misal: tethering HP).")

if __name__ == "__main__":
    main()