import asyncio
import json
import webbrowser
from fastmcp import Client
from fastmcp.client.auth import OAuth

MCP_URL = "https://mcp-resource.onrender.com/mcp"
CALLBACK_PORT = 8765
CALLBACK_URL = f"http://localhost:{CALLBACK_PORT}/callback"

async def custom_redirect(url: str):
    print("\n" + "=" * 70)
    print(">>> MICROSOFT LOGIN REQUIRED:")
    print("If your browser did not automatically open, copy & paste this link:")
    print(url)
    print("=" * 70 + "\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[Notice] Could not auto-launch browser ({e}). Please use the URL above.")

async def main():
    print("=" * 70)
    print("Connecting to live Resource MCP...")
    print(f"MCP Endpoint:  {MCP_URL}")
    print(f"Callback Port: {CALLBACK_PORT}")
    print(f"Callback URL:  {CALLBACK_URL}")
    print("=" * 70)

    auth = OAuth(
        mcp_url=MCP_URL,
        client_name="ATS app",
        callback_port=CALLBACK_PORT,
    )
    auth.redirect_handler = custom_redirect

    print("\n[Step 1] Initiating OAuth handshake with Microsoft...")
    async with Client(MCP_URL, auth=auth) as client:
        print("\n[Step 2] Authentication successful! Client connected.")

        # 1. WhoAmI
        print("\n--- 1. User Identity (whoami) ---")
        try:
            me = await client.call_tool("whoami")
            me_data = me.data if hasattr(me, "data") else me
            print(json.dumps(me_data, indent=2, default=str))
        except Exception as e:
            print(f"[Error] whoami failed: {e}")

        # 2. Search Files
        print("\n--- 2. Testing 'search_files' tool ---")
        found_paths = []
        for q in ["resume", "pdf", "Canada"]:
            try:
                s_res = await client.call_tool("search_files", {"query": q, "max_results": 10})
                s_data = s_res.data if hasattr(s_res, "data") else s_res
                items = s_data.get("items", []) if isinstance(s_data, dict) else (s_data if isinstance(s_data, list) else [])
                print(f"Search '{q}': returned {len(items)} items")
                for it in items[:5]:
                    p = it.get("path") or it.get("name")
                    print(f"  * [{it.get('type', 'item')}] {p}")
                    if p and p not in found_paths:
                        found_paths.append(p)
            except Exception as e:
                print(f"Search '{q}' error: {e}")

        # 3. List SharePoint Root Folder
        print("\n--- 3. Testing 'list_folder' tool (root: '') ---")
        try:
            top = await client.call_tool("list_folder", {"path": ""})
            top_data = top.data if hasattr(top, "data") else top
            print("Root folder listing:")
            print(json.dumps(top_data, indent=2, default=str))
        except Exception as e:
            print(f"[Error] list_folder root failed: {e}")

        # 4. If search found files/folders, test listing or getting info
        if found_paths:
            test_path = found_paths[0]
            print(f"\n--- 4. Testing 'get_file_info' on discovered item '{test_path}' ---")
            try:
                info = await client.call_tool("get_file_info", {"path": test_path})
                info_data = info.data if hasattr(info, "data") else info
                print(json.dumps(info_data, indent=2, default=str))
            except Exception as e:
                print(f"[Error] get_file_info failed: {e}")

    print("\n" + "=" * 70)
    print("Live MCP Verification Complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
