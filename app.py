import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# import spaces before any torch or transformers modules are loaded
try:
    import spaces
except ImportError:
    class MockSpaces:
        def GPU(self, fn=None, duration=None):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = MockSpaces()

try:
    import gradio as gr
    has_gradio = True
except ImportError:
    has_gradio = False

from app.server import app as fastapi_app, adapter

if has_gradio:
    @spaces.GPU
    def evaluate_candidate_gpu(job_description: str, resume_text: str):
        res = adapter.evaluate_candidate(
            job_title="Candidate Evaluation",
            job_description=job_description or "Software Engineer",
            resume_text=resume_text or "Python Developer"
        )
        import json
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

    # Disable SSR mode to prevent port 7861 binding conflicts on Hugging Face Spaces
    app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)
else:
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
