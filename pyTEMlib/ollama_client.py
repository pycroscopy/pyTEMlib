import asyncio
import json
import subprocess
import sys
from pathlib import Path
from ollama import chat

class MCPServer:
    def __init__(self):
        self.process = None

    async def start(self):
        """Start MCP server in background"""
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path("pyTEMlib/mcpserver.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Initialize the server
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ollama-client",
                    "version": "1.0"
                }
            }
        }

        self.process.stdin.write((json.dumps(init_request) + "\n").encode())
        await self.process.stdin.drain()

        # Read initialization response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode().strip())
        print(f"Server initialized: {response}")

        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }
        self.process.stdin.write((json.dumps(initialized_notification) + "\n").encode())
        await self.process.stdin.drain()

    async def call_tool(self, tool_name, arguments):
        """Call a tool on the MCP server"""
        if not self.process:
            raise Exception("Server not started")

        # Send tool call request
        call_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        self.process.stdin.write((json.dumps(call_request) + "\n").encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode().strip())

        if "error" in response:
            raise Exception(f"Tool call error: {response['error']}")

        return response["result"]["content"][0]["text"]

    async def stop(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            await self.process.wait()

def get_available_tools():
    """Get available tools from MCP server"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "hello",
                "description": "Greet someone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]
    return tools

async def main():
    server = MCPServer()
    await server.start()

    try:
        messages = [
            {
                "role": "user",
                "content": "Please greet Alice and add 5 + 3, then tell me the results"
            }
        ]

        while True:
            # Get LLM response
            response = chat(
                model="mistral",
                messages=messages,
                tools=get_available_tools(),
                stream=False
            )

            message = response['message']
            print(f"LLM: {message.get('content', '')}")

            # Check if LLM wants to use tools
            if 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    tool_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']  # Already a dict

                    print(f"Calling tool: {tool_name} with args: {arguments}")

                    # Call the actual MCP server tool
                    try:
                        result = await server.call_tool(tool_name, arguments)
                        print(f"Tool result: {result}")

                        # Add tool result to conversation
                        messages.append({
                            "role": "assistant",
                            "content": message.get('content', ''),
                            "tool_calls": message.get('tool_calls', [])
                        })
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.get('id')
                        })
                    except Exception as e:
                        print(f"Tool call failed: {e}")
                        break
            else:
                # No more tool calls, conversation complete
                break

    finally:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())