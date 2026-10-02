import requests
import json
from urllib.parse import urljoin

class RemoteMCPClient:
    """Direct HTTP client for MCP server with SSE support."""
    
    def __init__(self, ip_address: str, port: int = 8000):
        self.ip_address = ip_address
        self.port = port
        self.base_url = f"http://{ip_address}:{port}"
        self.request_id = 1
        self.active_endpoint = "/mcp"
        self.session_id = None
        self.server_info = None
        self._stream = None
        self._events = None
        self._message_url = None
    
    def _create_request(self, method: str, params: dict = None):
        """Create a JSON-RPC request."""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        self.request_id += 1
        return request
    
    def _open_sse_stream(self):
        """Open the SSE stream and discover its JSON-RPC message endpoint."""
        url = f"{self.base_url}{self.active_endpoint}"
        self._stream = requests.get(
            url,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=30,
        )
        self._stream.raise_for_status()
        self._events = iter(self._stream.iter_lines(decode_unicode=True))

        if next(self._events, None) != "event: endpoint":
            raise RuntimeError("MCP server did not return an SSE endpoint")

        endpoint = next(self._events, "").removeprefix("data: ").strip()
        if not endpoint:
            raise RuntimeError("MCP server returned an empty SSE endpoint")
        self._message_url = urljoin(url, endpoint)

    def _request(self, method: str, params: dict = None) -> dict:
        """Send JSON-RPC over the SSE transport and read its response event."""
        if self._message_url is None:
            self._open_sse_stream()

        request = self._create_request(method, params)
        response = requests.post(
            self._message_url,
            json=request,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()

        for line in self._events:
            if line and line.startswith("data:"):
                return json.loads(line.removeprefix("data: ").strip())

        raise RuntimeError("MCP SSE stream ended before returning a response")
    
    def initialize(self):
        """Initialize the connection using the server's SSE transport."""
        result = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "remote-client", "version": "1.0"}
        })
        self.server_info = result.get("result", {}).get("serverInfo", {})
        return result
    
    def list_tools(self):
        """List available tools on server."""
        if not self.server_info:
            raise Exception("Not initialized. Call initialize() first.")
        
        return self._request("tools/list")
    
    def call_tool(self, name: str, arguments: dict):
        """Call a specific tool."""
        if not self.server_info:
            raise Exception("Not initialized. Call initialize() first.")
        
        return self._request("tools/call", {
            "name": name,
            "arguments": arguments
        })

    def close(self):
        """Close the SSE stream."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._events = None
            self._message_url = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# Usage
if __name__ == "__main__":
    client = RemoteMCPClient("10.46.217.241", 9091)
    
    # Initialize
    init_response = client.initialize()
    print("\n\n exit()" \
    "Initialized:", init_response)
    
    # List tools
    tools = client.list_tools()
    print("\n\n Available tools:", tools)
    
    # Call a tool
    result = client.call_tool("CAMERA_Status", {})
    print("\n\n Tool result:", result)