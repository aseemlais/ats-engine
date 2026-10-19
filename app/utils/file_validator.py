from pathlib import Path
from typing import Tuple

class FileValidationError(Exception):
    pass

def validate_resume_file(file_path: Path | str, max_size_mb: int = 15) -> Tuple[bool, str]:
    """
    Validates resume file path, existence, size, and header signature.
    Returns (is_valid, error_message).
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File does not exist: {path}"
    
    if not path.is_file():
        return False, f"Path is not a regular file: {path}"
        
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        return False, "File is empty (0 bytes)"
        
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.2f} MB) exceeds maximum limit of {max_size_mb} MB"
        
    suffix = path.suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        return False, f"Unsupported file extension: {suffix}. Supported: .pdf, .docx, .txt"
        
    # Check magic bytes
    try:
        with open(path, "rb") as f:
            header = f.read(8)
            if suffix == ".pdf" and not header.startswith(b"%PDF"):
                return False, "Invalid PDF header signature (corrupted or renamed file)"
            elif suffix == ".docx" and not header.startswith(b"PK\x03\x04"):
                return False, "Invalid DOCX header signature (not a valid zip package)"
    except Exception as e:
        return False, f"Error reading file header: {str(e)}"
        
    return True, ""
