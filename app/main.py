from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.resume_routes import router as resume_router
from app.api.job_routes import router as job_router
from app.api.search_routes import router as search_router
from app.database.storage import MetadataStorage
from app.database.qdrant import QdrantVectorDB

app = FastAPI(
    title="Scalable Hybrid Semantic ATS",
    description="High-Speed Hybrid Information Retrieval & Candidate Ranking for 10,000+ Resumes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume_router, prefix="/api/v1")
app.include_router(job_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")

storage = MetadataStorage()
qdrant_db = QdrantVectorDB()

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/api/v1/health")
def health_check():
    """System health and status monitor."""
    return {
        "status": "healthy",
        "service": "Scalable Hybrid Semantic ATS",
        "indexed_resumes": storage.count_resumes(),
        "vector_db_points": qdrant_db.count_resumes()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
