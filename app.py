import os
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
from app.server import app as fastapi_app

# Create a clean status dashboard for visitors on Hugging Face
with gr.Blocks(title="Hybrid Semantic ATS Engine") as demo:
    gr.Markdown("# ⚡ Scalable Hybrid Semantic ATS Engine")
    gr.Markdown(
        "This Space provides the high-performance NLP & Machine Learning retrieval "
        "and candidate scoring backend for the ATS Application."
    )
    with gr.Row():
        gr.Markdown("### ✅ System Status: **Online & Ready**")
    with gr.Accordion("API Endpoints", open=True):
        gr.Markdown(
            "- `POST /invocations`: MLflow ResponsesAgent candidate evaluation & ranking\n"
            "- `POST /api/files/upload`: PyMuPDF layout-aware resume parsing\n"
            "- `GET /health`: Engine health check\n"
            "- `GET /docs`: OpenAPI interactive documentation"
        )

# Mount Gradio onto the existing FastAPI application at /status so / is free or at /
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
