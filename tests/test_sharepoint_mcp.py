import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.server import app
from app.services.mcp_client import mcp_client_service, MockSharePointProvider

@pytest.fixture
def client():
    with patch("app.server.extraction_service.start"), \
         patch("app.server.extraction_service.stop"), \
         patch("app.server.is_supabase_configured", return_value=False):
        yield TestClient(app, raise_server_exceptions=False)

@pytest.mark.anyio
async def test_mock_sharepoint_provider_listing():
    provider = MockSharePointProvider()
    res = await provider.list_folder("")
    assert res.count >= 3
    names = [it.name for it in res.items]
    assert "Repository" in names
    assert "Canada opportunity JD" in names

    # Subfolder listing
    repo_res = await provider.list_folder("Repository")
    assert repo_res.current_path == "Repository"
    assert repo_res.parent_path == ""
    repo_names = [it.name for it in repo_res.items]
    assert ".Net Developer" in repo_names
    assert "AI Engineer" in repo_names
    assert "AWS" in repo_names

@pytest.mark.anyio
async def test_mock_sharepoint_provider_file_content():
    provider = MockSharePointProvider()
    filename, content, mime = await provider.get_file_content("Repository/AI Engineer/Arjun_Mehta_Lead_AI_Engineer.txt")
    assert filename == "Arjun_Mehta_Lead_AI_Engineer.txt"
    assert len(content) > 100
    assert mime == "text/plain"
    assert b"Arjun Mehta" in content
    assert b"PyTorch" in content

def test_sharepoint_status_endpoint(client):
    response = client.get("/api/sharepoint/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "mock_mode" in data
    assert data["identity"]["authenticated"] is True

def test_sharepoint_folders_endpoint(client):
    # Root folder
    response = client.get("/api/sharepoint/folders")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["count"] >= 3
    assert any(it["name"] == "Repository" for it in data["items"])

    # Subfolder
    response_sub = client.get("/api/sharepoint/folders?path=Repository")
    assert response_sub.status_code == 200
    sub_data = response_sub.json()
    assert sub_data["current_path"] == "Repository"
    assert any(it["name"] == "AI Engineer" for it in sub_data["items"])

def test_sharepoint_import_endpoint(client):
    payload = {
        "paths": [
            "Repository/AI Engineer/Arjun_Mehta_Lead_AI_Engineer.txt"
        ]
    }
    response = client.post("/api/sharepoint/import", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["successful"] == 1
    assert data["failed"] == 0

    item = data["items"][0]
    assert item["success"] is True
    assert item["filename"] == "Arjun_Mehta_Lead_AI_Engineer.txt"
    assert item.get("resumeText") is not None
    assert "PyTorch" in item.get("resumeText")

def test_sharepoint_fetch_resumes_endpoint(client):
    payload = {
        "sharePointLink": "https://datapattern.sharepoint.com/sites/hr/Repository/AI Engineer"
    }
    response = client.post("/api/sharepoint/fetch-resumes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "resumes" in data
    assert data["count"] >= 1
    names = [r["fileName"] for r in data["resumes"]]
    assert any("Arjun" in n for n in names)

def test_mcp_oauth_callback_endpoint(client):
    response = client.get("/api/mcp/callback?code=mock_auth_code_12345&state=xyz")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Authentication Successful" in response.text
    assert "MCP_AUTH_SUCCESS" in response.text
