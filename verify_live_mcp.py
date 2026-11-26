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
        print("\n--- 1. Testing 'whoami' tool ---")
        try:
            me = await client.call_tool("whoami")
            me_data = me.data if hasattr(me, "data") else me
            print(json.dumps(me_data, indent=2, default=str))
        except Exception as e:
            print(f"[Error] whoami failed: {e}")

        # 2. List Tools
        print("\n--- 2. Discovering Available MCP Tools ---")
        try:
            tools = await client.list_tools()
            tool_list = tools if isinstance(tools, list) else getattr(tools, "tools", [])
            print(f"Discovered {len(tool_list)} tools:")
            for t in tool_list:
                name = getattr(t, "name", str(t))
                desc = getattr(t, "description", "")
                schema = getattr(t, "parameters", getattr(t, "inputSchema", {}))
                print(f"\n  Tool: {name}")
                if desc:
                    print(f"    Description: {desc}")
                if schema:
                    print(f"    Parameters: {json.dumps(schema, default=str)}")
        except Exception as e:
            print(f"[Error] list_tools failed: {e}")

        # 3. List SharePoint Root Folder
        print("\n--- 3. Testing 'list_folder' tool ---")
        try:
            top = await client.call_tool("list_folder", {"path": ""})
            top_data = top.data if hasattr(top, "data") else top
            print(json.dumps(top_data, indent=2, default=str))
        except Exception as e:
            print(f"[Error] list_folder failed: {e}")

    print("\n" + "=" * 70)
    print("Live MCP Verification Complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
