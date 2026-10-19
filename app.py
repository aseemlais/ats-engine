import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure spaces module exists for local compatibility
if "spaces" not in sys.modules:
    try:
        import spaces
    except ImportError:
        import types
        mock_spaces = types.ModuleType("spaces")
        mock_spaces.GPU = lambda fn=None, duration=None: fn if callable(fn) else (lambda f: f)
        sys.modules["spaces"] = mock_spaces

import spaces
import gradio as gr
from app.server import app as fastapi_app, adapter

@spaces.GPU(duration=60)
def evaluate_candidate_gpu(job_description: str, resume_text: str):
    res = adapter.evaluate_candidate(
        job_title="Candidate Evaluation",
        job_description=job_description or "Software Engineer",
        resume_text=resume_text or "Python Developer"
    )
    return json.dumps(res, indent=2)

demo = gr.Interface(
    fn=evaluate_candidate_gpu,
    inputs=[
        gr.Textbox(label="Job Description", lines=4, placeholder="Enter job description..."),
        gr.Textbox(label="Resume Text", lines=4, placeholder="Enter resume text...")
    ],
    outputs=gr.Code(label="Screening Output (JSON)", language="json"),
    title="⚡ Scalable Hybrid Semantic ATS Engine",
    description="ZeroGPU-accelerated candidate screening and ranking backend."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=True,
        ssr_mode=False
    )
    # Expose all ATS FastAPI routes on the running server
    for route in fastapi_app.routes:
        app.routes.append(route)
    demo.block_thread()
