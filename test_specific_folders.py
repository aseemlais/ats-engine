"""
Diagnostic Script to test specific Canada 🍁 and Repository folder paths
against the live Resource MCP Server (https://mcp-resource.onrender.com/mcp).
"""

import asyncio
import json
from fastmcp import Client
from fastmcp.client.auth import OAuth

MCP_URL = "https://mcp-resource.onrender.com/mcp"
CALLBACK_PORT = 8765

# Specific target paths from Microsoft Teams channel
TARGET_PATHS = [
    "Repository",
    "Repository/AI Engineer",
    "Repository/.Net Developer",
    "Repository/AWS",
    "Repository/Azure",
    "Repository/Business Analyst",
    "Canada 🍁/Repository",
    "Canada/Repository",
    "Canada opportunity JD",
    "Capgemini",
]

async def main():
    print("=" * 70)
    print("Testing Canada 🍁 & Repository Paths on Live Resource MCP")
    print(f"MCP URL:       {MCP_URL}")
    print(f"Callback Port: {CALLBACK_PORT}")
    print("=" * 70)

    auth = OAuth(mcp_url=MCP_URL, client_name="ATS App", callback_port=CALLBACK_PORT)
    async with Client(MCP_URL, auth=auth) as client:
        print("\n[Step 1] Verifying signed-in account identity...")
        who = await client.call_tool("whoami")
        who_data = who.data if hasattr(who, "data") else who
        print(f"Logged in as: {who_data}")

        print("\n[Step 2] Testing specific folder paths with 'list_folder'...")
        discovered_files = []

        for folder_path in TARGET_PATHS:
            print(f"\n--- Testing Path: '{folder_path}' ---")
            try:
                res = await client.call_tool("list_folder", {"path": folder_path})
                data = res.data if hasattr(res, "data") else res

                if isinstance(data, str):
                    if "error" in data.lower() or "did not allow" in data.lower():
                        print(f"  [SharePoint Access Denied] {data}")
                        continue
                    try:
                        data = json.loads(data)
                    except Exception:
                        print(f"  [Response String] {data}")
                        continue

                items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                print(f"  [SUCCESS] Found {len(items)} items in '{folder_path}':")
                for it in items[:6]:
                    item_name = it.get("name") or it.get("path")
                    item_type = it.get("type", "file")
                    print(f"    - [{item_type}] {item_name}")
                    if item_type == "file":
                        discovered_files.append(it.get("path") or item_name)
            except Exception as e:
                print(f"  [Failed] {e}")

        # Test read_file if any file was discovered
        print("\n[Step 3] Testing 'read_file' tool...")
        test_file = discovered_files[0] if discovered_files else "Repository/AI Engineer/Arjun_Mehta_Lead_AI_Engineer.txt"
        print(f"Attempting to read file: '{test_file}'")
        try:
            read_res = await client.call_tool("read_file", {
                "path": test_file,
                "start_char": 0,
                "max_chars": 500,
            })
            read_data = read_res.data if hasattr(read_res, "data") else read_res
            if isinstance(read_data, dict):
                content = read_data.get("text") or read_data.get("content") or ""
                print(f"  [SUCCESS] Read {len(content)} characters:")
                print(f"  Preview: {content[:200]}...")
            else:
                print(f"  [Result] {str(read_data)[:200]}")
        except Exception as e:
            print(f"  [read_file Note] {e}")

    print("\n" + "=" * 70)
    print("Verification Completed!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
