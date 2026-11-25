"""
Model Context Protocol (MCP) Client Service for Microsoft 365 / SharePoint Integration.

Supports:
- Swappable provider architecture: Mock provider for offline testing vs Live FastMCP provider.
- Dynamic fallback: Gracefully handles environments without fastmcp or without OAuth whitelist.
- End-to-end resume stream ingestion directly into ATS parser & Supabase repository.
"""

import abc
import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mcp_client")


class SharePointItem:
    """Represents a file or folder in SharePoint."""

    def __init__(
        self,
        name: str,
        path: str,
        item_type: str,  # "file" or "folder"
        size: Optional[int] = None,
        last_modified: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        self.name = name
        self.path = path.strip("/")
        self.item_type = item_type
        self.size = size or 0
        self.last_modified = last_modified or datetime.now(timezone.utc).isoformat()
        self.extension = extension or (Path(name).suffix.lower() if "." in name else "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "type": self.item_type,
            "size": self.size,
            "last_modified": self.last_modified,
            "extension": self.extension,
        }


class SharePointFolderResponse:
    """Structured response for a SharePoint folder listing."""

    def __init__(
        self,
        current_path: str,
        parent_path: Optional[str],
        items: List[SharePointItem],
        total_count: int,
    ):
        self.current_path = current_path
        self.parent_path = parent_path
        self.items = items
        self.count = total_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_path": self.current_path,
            "parent_path": self.parent_path,
            "count": self.count,
            "items": [item.to_dict() for item in self.items],
        }


class BaseSharePointProvider(abc.ABC):
    """Abstract interface for SharePoint providers."""

    @abc.abstractmethod
    async def list_folder(self, path: str = "") -> SharePointFolderResponse:
        """List files and subfolders at the specified path."""
        pass

    @abc.abstractmethod
    async def get_file_content(self, path: str) -> Tuple[str, bytes, str]:
        """Download file content. Returns (filename, raw_bytes, mime_type)."""
        pass

    @abc.abstractmethod
    async def whoami(self) -> Dict[str, Any]:
        """Check authentication status and return signed-in identity."""
        pass


class MockSharePointProvider(BaseSharePointProvider):
    """
    In-memory mock SharePoint provider for isolated local testing.
    Provides a realistic directory tree with valid sample resume content.
    """

    SAMPLE_RESUMES = {
        "Engineering/Alex_Rivera_Staff_Software_Engineer.txt": (
            "Alex Rivera\n"
            "Staff Software Engineer\n"
            "Email: alex.rivera@example.com | Phone: +1-415-555-0199\n\n"
            "Professional Summary:\n"
            "Seasoned software engineer with 8+ years experience building cloud-native microservices, "
            "distributed pipelines, and high-throughput backend APIs.\n\n"
            "Technical Skills:\n"
            "Python, FastAPI, Django, Docker, Kubernetes, AWS, PostgreSQL, Redis, Apache Kafka, CI/CD\n\n"
            "Work Experience:\n"
            "Staff Backend Engineer - CloudScale Inc (2021 - Present)\n"
            "- Architected multi-region event streaming platform handling 20,000 req/sec.\n"
            "- Migrated monolithic workloads into containerized Kubernetes pods.\n"
            "Senior Software Developer - DataCore Labs (2018 - 2021)\n"
            "- Designed REST APIs with FastAPI and PostgreSQL.\n\n"
            "Education:\n"
            "B.S. in Computer Science, University of California, Berkeley (2018)\n"
        ),
        "Engineering/Priya_Sharma_Senior_FullStack_Engineer.txt": (
            "Priya Sharma\n"
            "Senior Full Stack Engineer\n"
            "Email: priya.sharma@example.com | Phone: +1-647-555-0142\n\n"
            "Summary:\n"
            "Full Stack Developer with 6 years of experience in React, TypeScript, Node.js, and Python.\n\n"
            "Skills:\n"
            "React, TypeScript, Next.js, Node.js, Express, Python, TailwindCSS, MongoDB, Docker, Git\n\n"
            "Experience:\n"
            "Senior Full Stack Developer - WebFlow Solutions (2020 - Present)\n"
            "- Built recruiter dashboards in Next.js and TypeScript.\n"
            "- Implemented real-time collaboration features using WebSockets.\n\n"
            "Education:\n"
            "Bachelor of Technology in Information Technology (2019)\n"
        ),
        "Engineering/Backend Candidates/David_Chen_Principal_Systems_Engineer.txt": (
            "David Chen\n"
            "Principal Systems Engineer\n"
            "Email: david.chen@example.com | Phone: +1-206-555-0188\n\n"
            "Summary:\n"
            "Systems architect with 10+ years specializing in distributed systems, Go, C++, Linux kernel performance, "
            "and Kubernetes platform engineering.\n\n"
            "Skills:\n"
            "Go, C++, Linux, Kubernetes, gRPC, Distributed Systems, Terraform, Grafana, Prometheus\n\n"
            "Experience:\n"
            "Principal Engineer - HyperScale Systems (2019 - Present)\n"
            "- Led infrastructure engineering team of 12 developers.\n\n"
            "Education:\n"
            "M.S. in Computer Engineering, University of Washington (2015)\n"
        ),
        "Data Science & AI/Arjun_Mehta_Lead_ML_Engineer.txt": (
            "Arjun Mehta\n"
            "Lead Machine Learning Engineer\n"
            "Email: arjun.mehta@example.com | Phone: +1-312-555-0177\n\n"
            "Summary:\n"
            "Machine Learning Engineer with 7+ years delivering NLP, transformer architectures, LLM fine-tuning, "
            "and retrieval augmented generation pipelines.\n\n"
            "Skills:\n"
            "Python, PyTorch, Hugging Face, Transformers, SentenceTransformers, Vector Databases, Qdrant, MLOps\n\n"
            "Experience:\n"
            "Lead ML Engineer - NeuralAI Corp (2021 - Present)\n"
            "- Deployed semantic search models using sentence-transformers and FAISS.\n\n"
            "Education:\n"
            "M.S. in Artificial Intelligence, Carnegie Mellon University (2018)\n"
        ),
    }

    def __init__(self):
        # Build hierarchy tree
        self._tree: Dict[str, List[SharePointItem]] = {
            "": [
                SharePointItem(name="Engineering", path="Engineering", item_type="folder"),
                SharePointItem(name="Product & Design", path="Product & Design", item_type="folder"),
                SharePointItem(name="Data Science & AI", path="Data Science & AI", item_type="folder"),
                SharePointItem(
                    name="Company_Hiring_Guidelines_2026.docx",
                    path="Company_Hiring_Guidelines_2026.docx",
                    item_type="file",
                    size=14200,
                ),
            ],
            "Engineering": [
                SharePointItem(name="Backend Candidates", path="Engineering/Backend Candidates", item_type="folder"),
                SharePointItem(
                    name="Alex_Rivera_Staff_Software_Engineer.txt",
                    path="Engineering/Alex_Rivera_Staff_Software_Engineer.txt",
                    item_type="file",
                    size=len(self.SAMPLE_RESUMES["Engineering/Alex_Rivera_Staff_Software_Engineer.txt"].encode("utf-8")),
                ),
                SharePointItem(
                    name="Priya_Sharma_Senior_FullStack_Engineer.txt",
                    path="Engineering/Priya_Sharma_Senior_FullStack_Engineer.txt",
                    item_type="file",
                    size=len(self.SAMPLE_RESUMES["Engineering/Priya_Sharma_Senior_FullStack_Engineer.txt"].encode("utf-8")),
                ),
            ],
            "Engineering/Backend Candidates": [
                SharePointItem(
                    name="David_Chen_Principal_Systems_Engineer.txt",
                    path="Engineering/Backend Candidates/David_Chen_Principal_Systems_Engineer.txt",
                    item_type="file",
                    size=len(self.SAMPLE_RESUMES["Engineering/Backend Candidates/David_Chen_Principal_Systems_Engineer.txt"].encode("utf-8")),
                ),
            ],
            "Product & Design": [],
            "Data Science & AI": [
                SharePointItem(
                    name="Arjun_Mehta_Lead_ML_Engineer.txt",
                    path="Data Science & AI/Arjun_Mehta_Lead_ML_Engineer.txt",
                    item_type="file",
                    size=len(self.SAMPLE_RESUMES["Data Science & AI/Arjun_Mehta_Lead_ML_Engineer.txt"].encode("utf-8")),
                ),
            ],
        }

    async def list_folder(self, path: str = "") -> SharePointFolderResponse:
        clean_path = path.strip("/")
        items = self._tree.get(clean_path, [])
        parent_path = None
        if clean_path:
            parent_path = str(Path(clean_path).parent).replace("\\", "/")
            if parent_path == ".":
                parent_path = ""

        return SharePointFolderResponse(
            current_path=clean_path,
            parent_path=parent_path,
            items=items,
            total_count=len(items),
        )

    async def get_file_content(self, path: str) -> Tuple[str, bytes, str]:
        clean_path = path.strip("/")
        filename = Path(clean_path).name

        if clean_path in self.SAMPLE_RESUMES:
            text = self.SAMPLE_RESUMES[clean_path]
            return filename, text.encode("utf-8"), "text/plain"

        # Generate standard placeholder resume for any other requested file path
        fallback_text = (
            f"Candidate: {filename.replace('.txt', '').replace('.pdf', '').replace('_', ' ')}\n"
            f"Role: Software Engineer\n"
            f"Email: candidate.{abs(hash(clean_path)) % 1000}@example.com\n"
            f"Skills: Python, React, SQL, Git, Linux, Docker\n"
            f"Experience: 4 years as a Full Stack Developer.\n"
        )
        return filename, fallback_text.encode("utf-8"), "text/plain"

    async def whoami(self) -> Dict[str, Any]:
        return {
            "authenticated": True,
            "mode": "mock",
            "account": "mock.recruiter@datapattern.ai",
            "display_name": "Mock DataPattern Recruiter",
            "source": "Local Mock SharePoint Provider",
        }


class LiveFastMCPProvider(BaseSharePointProvider):
    """
    Live FastMCP provider connecting to the remote MCP server hosted on Render.
    Uses fastmcp Client and OAuth flow with ALLOWED_CLIENT_REDIRECTS.
    """

    def __init__(
        self,
        mcp_url: Optional[str] = None,
        callback_port: int = 8765,
    ):
        self.mcp_url = mcp_url or os.getenv("MCP_RESOURCE_URL", "https://mcp-resource.onrender.com/mcp")
        self.callback_port = int(os.getenv("MCP_CALLBACK_PORT", str(callback_port)))
        self._auth = None

    def _ensure_fastmcp(self):
        try:
            import fastmcp
            return fastmcp
        except ImportError:
            raise RuntimeError(
                "The 'fastmcp' library is required for live MCP integration. "
                "Install it using: pip install fastmcp"
            )

    async def whoami(self) -> Dict[str, Any]:
        self._ensure_fastmcp()
        from fastmcp import Client
        from fastmcp.client.auth import OAuth

        auth = OAuth(mcp_url=self.mcp_url, client_name="ATS App", callback_port=self.callback_port)
        async with Client(self.mcp_url, auth=auth) as client:
            res = await client.call_tool("whoami")
            return {
                "authenticated": True,
                "mode": "live",
                "data": res.data if hasattr(res, "data") else res,
            }

    async def list_folder(self, path: str = "") -> SharePointFolderResponse:
        self._ensure_fastmcp()
        from fastmcp import Client
        from fastmcp.client.auth import OAuth

        auth = OAuth(mcp_url=self.mcp_url, client_name="ATS App", callback_port=self.callback_port)
        async with Client(self.mcp_url, auth=auth) as client:
            res = await client.call_tool("list_folder", {"path": path})
            data = res.data if hasattr(res, "data") else (res or {})
            raw_items = data.get("items", [])

            items: List[SharePointItem] = []
            for item in raw_items:
                items.append(
                    SharePointItem(
                        name=Path(item.get("path", "")).name or item.get("name", "Unknown"),
                        path=item.get("path", ""),
                        item_type=item.get("type", "file"),
                        size=item.get("size", 0),
                        last_modified=item.get("last_modified"),
                    )
                )

            clean_path = path.strip("/")
            parent = str(Path(clean_path).parent).replace("\\", "/") if clean_path else None
            if parent == ".":
                parent = ""

            return SharePointFolderResponse(
                current_path=clean_path,
                parent_path=parent,
                items=items,
                total_count=len(items),
            )

    async def get_file_content(self, path: str) -> Tuple[str, bytes, str]:
        self._ensure_fastmcp()
        from fastmcp import Client
        from fastmcp.client.auth import OAuth

        filename = Path(path).name
        auth = OAuth(mcp_url=self.mcp_url, client_name="ATS App", callback_port=self.callback_port)
        async with Client(self.mcp_url, auth=auth) as client:
            # Fallback tool discovery: try download_file, then read_file, then get_file
            tool_name = os.getenv("MCP_DOWNLOAD_TOOL_NAME", "download_file")
            try:
                res = await client.call_tool(tool_name, {"path": path})
            except Exception as e:
                logger.warning("Primary download tool %s failed (%s), trying read_file", tool_name, e)
                res = await client.call_tool("read_file", {"path": path})

            data = res.data if hasattr(res, "data") else res
            # Handle possible return shapes: raw bytes, base64 dict, or text
            if isinstance(data, bytes):
                return filename, data, "application/octet-stream"
            elif isinstance(data, dict):
                import base64
                if "content_base64" in data:
                    b = base64.b64decode(data["content_base64"])
                    return filename, b, data.get("content_type", "application/pdf")
                elif "text" in data:
                    return filename, data["text"].encode("utf-8"), "text/plain"
            elif isinstance(data, str):
                return filename, data.encode("utf-8"), "text/plain"

            raise ValueError(f"Unrecognized content structure returned by MCP for {path}")


class MCPClientService:
    """
    Manager that provides active SharePoint provider according to configuration.
    Defaults to MockSharePointProvider when MCP_MOCK_MODE is true or fastmcp is absent.
    """

    def __init__(self):
        self._mock_provider = MockSharePointProvider()
        self._live_provider: Optional[LiveFastMCPProvider] = None

    @property
    def is_mock_mode(self) -> bool:
        forced = os.getenv("MCP_MOCK_MODE", "").strip().lower()
        if forced in ("true", "1", "yes", "on"):
            return True
        if forced in ("false", "0", "no", "off"):
            return False

        # Default to mock mode if fastmcp is not installed
        try:
            import fastmcp  # noqa: F401
            return False
        except ImportError:
            return True

    def get_provider(self) -> BaseSharePointProvider:
        if self.is_mock_mode:
            return self._mock_provider

        if self._live_provider is None:
            self._live_provider = LiveFastMCPProvider()
        return self._live_provider


# Global singleton instance
mcp_client_service = MCPClientService()
