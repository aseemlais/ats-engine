"""
Diagnostic script to inspect account identity, permissions, and test search_files
against the live Resource MCP Server (https://mcp-resource.onrender.com/mcp).
"""

import asyncio
import json
from fastmcp import Client
from fastmcp.client.auth import OAuth

MCP_URL = "https://mcp-resource.onrender.com/mcp"
CALLBACK_PORT = 8765

async def main():
    print("=" * 70)
    print("SharePoint Access Diagnostic Tool")
    print(f"MCP URL: {MCP_URL}")
    print("=" * 70)

    auth = OAuth(mcp_url=MCP_URL, client_name="ATS App", callback_port=CALLBACK_PORT)
    async with Client(MCP_URL, auth=auth) as client:
        # Step 1: Whoami
        print("\n--- 1. Checking Currently Signed-In Identity ---")
        try:
            who = await client.call_tool("whoami")
            who_data = who.data if hasattr(who, "data") else who
            print("Signed in identity returned by Microsoft:")
            print(json.dumps(who_data, indent=2, default=str))
        except Exception as e:
            print(f"whoami failed: {e}")

        # Step 2: Test search_files
        print("\n--- 2. Testing 'search_files' across multiple queries ---")
        search_queries = ["Canada", "Repository", "Calling Log", "Developer", "Engineer", "pdf", "docx"]
        accessible_items = []

        for q in search_queries:
            try:
                s_res = await client.call_tool("search_files", {"query": q, "max_results": 5})
                s_data = s_res.data if hasattr(s_res, "data") else s_res
                items = s_data.get("items", []) if isinstance(s_data, dict) else (s_data if isinstance(s_data, list) else [])
                print(f"  Search query '{q}': found {len(items)} items")
                for it in items:
                    item_path = it.get("path") or it.get("name")
                    print(f"    * [{it.get('type')}] {item_path}")
                    if item_path and item_path not in accessible_items:
                        accessible_items.append(item_path)
            except Exception as e:
                print(f"  Search query '{q}' error: {e}")

        # Step 3: If any accessible items were found, test get_file_info or read_file
        if accessible_items:
            print(f"\n--- 3. Testing 'get_file_info' on accessible item: '{accessible_items[0]}' ---")
            try:
                info = await client.call_tool("get_file_info", {"path": accessible_items[0]})
                info_data = info.data if hasattr(info, "data") else info
                print(json.dumps(info_data, indent=2, default=str))
            except Exception as e:
                print(f"get_file_info error: {e}")
        else:
            print("\n[Summary] No items were returned by search_files. The signed-in account cannot query items in this library.")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
