import asyncio
import subprocess
import json
import sys
from pathlib import Path

async def main():
    # Start the MCP server as a subprocess
    server_process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path("pyTEMlib/mcpserver.py")),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0"
                }
            }
        }
        
        server_process.stdin.write((json.dumps(init_request) + "\n").encode())
        await server_process.stdin.drain()
        
        # Read initialization response
        response = await server_process.stdout.readline()
        print("Server response:", response.decode().strip())
        
        # Send list_tools request
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        server_process.stdin.write((json.dumps(list_tools_request) + "\n").encode())
        await server_process.stdin.drain()
        
        # Read tools response
        response = await server_process.stdout.readline()
        print("\nAvailable tools:", response.decode().strip())
        
        # Call hello tool
        call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "hello",
                "arguments": {"name": "Alice"}
            }
        }
        
        server_process.stdin.write((json.dumps(call_request) + "\n").encode())
        await server_process.stdin.drain()
        
        # Read tool call response
        response = await server_process.stdout.readline()
        print("\nhello('Alice') response:", response.decode().strip())
        
    finally:
        server_process.terminate()
        await server_process.wait()

if __name__ == "__main__":
    asyncio.run(main())