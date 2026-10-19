import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path | str) -> str:
    """Computes SHA-256 hash of a file for exact duplicate binary detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_text_hash(text: str) -> str:
    """Computes SHA-256 hash of normalized text for duplicate content detection."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
