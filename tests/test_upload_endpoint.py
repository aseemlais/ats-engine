"""MANUAL script (not an automated test): uploads a local PDF to a running server.
Usage: python tests/test_upload_endpoint.py "<path-to.pdf>"   (server must be running)
The previous version executed at import time with a hard-coded C:\\Users path, which broke test collection."""
import sys

def main(pdf_path):
    import urllib.request
    import json
    import os

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    filename = os.path.basename(pdf_path)
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + pdf_bytes + footer

    req = urllib.request.Request(
        "http://localhost:8000/api/files/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
        print("Upload status:", r.status)
        print("Success:", data.get("success"))
        print("Extracted character count:", len(data.get("resumeText", "")))
        print("Preview:\n", data.get("resumeText", "")[:150])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python tests/test_upload_endpoint.py <resume.pdf>")
    main(sys.argv[1])
