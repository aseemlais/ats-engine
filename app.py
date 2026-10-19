import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Mock spaces for local execution when HF spaces is not installed
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
    def screen_candidate(job_title: str, job_description: str, resume_text: str):
        return adapter.evaluate_candidate(
            job_title=job_title or "Position",
            job_description=job_description,
            resume_text=resume_text
        )

    with gr.Blocks(title="Hybrid Semantic ATS Engine") as demo:
        gr.Markdown("# ⚡ Scalable Hybrid Semantic ATS Engine")
        gr.Markdown("ZeroGPU-accelerated candidate screening and ranking API")
        with gr.Row():
            jd_box = gr.Textbox(label="Job Description", lines=3, placeholder="Paste JD here...")
            res_box = gr.Textbox(label="Resume Text", lines=3, placeholder="Paste Resume here...")
        eval_btn = gr.Button("Evaluate Candidate", variant="primary")
        out_json = gr.JSON(label="Evaluation Result")
        eval_btn.click(
            fn=screen_candidate,
            inputs=[gr.Textbox(value="Position", visible=False), jd_box, res_box],
            outputs=out_json
        )

    # Disable SSR mode to prevent port 7861 binding conflicts on Hugging Face Spaces
    app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)
else:
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
