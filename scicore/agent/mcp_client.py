"""MCP Client — connect SciCoreAgent to external MCP servers.

Implements the MCP client side of the Model Context Protocol (MCP)
so SciCoreAgent can consume tools from external MCP servers.

Architecture:
    MCPClient — manages one subprocess connection to an external MCP server.
                 Handles JSON-RPC 2.0 over stdin/stdout.

    MCPClientProvider — wraps one or more MCPClient instances into the
                        MCPProvider protocol expected by the ExtensionRegistry.

    start_configured_mcp_servers() — reads .scicore/mcp_servers.json,
                                     spawns each configured server, and
                                     registers them in the global registry.

Usage (called automatically by Agent.__init__):
    from .mcp_client import start_configured_mcp_servers
    start_configured_mcp_servers()
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .extensions import GLOBAL_EXTENSION_REGISTRY, MCPProvider
from .mcp_config import list_mcp_servers, MCPServerConfig

logger = logging.getLogger(__name__)

# MCP JSON-RPC protocol version
MCP_PROTOCOL_VERSION = "2025-06-18"

# Timeout (seconds) for initialize handshake.
INIT_TIMEOUT = 15
# Timeout (seconds) for individual tool calls.
CALL_TIMEOUT = 120


class MCPClientError(Exception):
    """Raised when MCP client operations fail."""


class MCPClient:
    """Manages a subprocess connection to one external MCP server.

    Handles the full lifecycle:
    1. Spawn the server process (command + args from config).
    2. Send `initialize` request, receive capabilities.
    3. Send `tools/list` request, cache tool schemas.
    4. Route `tools/call` requests during agent execution.
    5. Clean up subprocess on shutdown.
    """

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._process: subprocess.Popen | None = None
        self._tools: list[dict] = []
        self._server_info: dict = {}
        self._lock = threading.Lock()
        self._req_id = 0
        self._alive = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def alive(self) -> bool:
        return self._alive and self._process is not None and self._process.poll() is None

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def start(self) -> None:
        """Spawn the MCP server subprocess and complete the initialize handshake."""
        if self._alive:
            return

        cmd = [self._config.command] + list(self._config.args)

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except FileNotFoundError:
            raise MCPClientError(
                f"MCP server '{self._config.name}': command not found: {self._config.command}"
            )
        except Exception as e:
            raise MCPClientError(
                f"MCP server '{self._config.name}': failed to start process: {e}"
            )

        try:
            # Step 1 — initialize
            init_resp = self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "clientInfo": {"name": "scicore.bio", "version": "1.0.0"},
                    "capabilities": {},
                },
                timeout=INIT_TIMEOUT,
            )
            self._server_info = init_resp.get("result", init_resp).get("serverInfo", {})

            # Send initialized notification
            self._send_notification("notifications/initialized", {})

            # Step 2 — list tools
            tools_resp = self._request("tools/list", {}, timeout=INIT_TIMEOUT)
            raw_tools = tools_resp.get("result", tools_resp).get("tools", [])

            # Namespace tool names: mcp.<server_name>.<tool_name>
            self._tools = []
            for tool in raw_tools:
                namespaced = dict(tool)
                original_name = tool.get("name", "unknown")
                namespaced["name"] = f"mcp_{self._config.name}_{original_name}"
                namespaced["_mcp_server"] = self._config.name
                namespaced["_mcp_original_name"] = original_name
                self._tools.append(namespaced)

            self._alive = True
            logger.info(
                "MCP server '%s' started: %d tools available (server: %s %s)",
                self._config.name,
                len(self._tools),
                self._server_info.get("name", "?"),
                self._server_info.get("version", "?"),
            )

        except MCPClientError:
            self.stop()
            raise
        except Exception as e:
            self.stop()
            raise MCPClientError(
                f"MCP server '{self._config.name}': handshake failed: {e}"
            )

    def stop(self) -> None:
        """Terminate the subprocess and clean up."""
        self._alive = False
        self._tools.clear()
        if self._process:
            try:
                self._process.stdin.close()
                self._process.stdout.close()  # type: ignore
                self._process.stderr.close()  # type: ignore
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    # -- MCPProvider-compatible interface ------------------------------------

    def tool_schemas(self) -> list[dict]:
        """Return cached tool schemas (namespaced) for the agent."""
        if not self._alive:
            return []
        return list(self._tools)

    def call_tool(self, namespaced_name: str, arguments: dict) -> str | None:
        """Call a tool on this MCP server.

        *namespaced_name* is the full name including the mcp_<server>_ prefix.
        Returns the text content from the MCP response, or an error string.
        Returns None if this tool doesn't belong to this server.
        """
        prefix = f"mcp_{self._config.name}_"
        if not namespaced_name.startswith(prefix):
            return None  # not ours

        original_name = namespaced_name[len(prefix):]

        if not self._alive:
            return f"Error: MCP server '{self._config.name}' is not running."

        try:
            resp = self._request(
                "tools/call",
                {"name": original_name, "arguments": arguments},
                timeout=CALL_TIMEOUT,
            )
            result = resp.get("result", resp)
            content = result.get("content", [])
            is_error = result.get("isError", False)

            # Extract text from content array.
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)

            output = "\n".join(texts)
            if is_error:
                return f"Error from mcp_{self._config.name}/{original_name}: {output}"
            return output

        except MCPClientError as e:
            return f"Error calling mcp_{self._config.name}/{original_name}: {e}"

    # -- JSON-RPC helpers ----------------------------------------------------

    def _next_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def _request(self, method: str, params: dict, timeout: float = 60) -> dict:
        """Send a JSON-RPC request and return the parsed response."""
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        self._send(payload)
        return self._recv(timeout)

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send(payload)

    def _send(self, payload: dict) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        if not self._process or not self._process.stdin:
            raise MCPClientError(f"MCP server '{self._config.name}' is not running.")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._alive = False
            raise MCPClientError(
                f"MCP server '{self._config.name}': write failed (process may have crashed): {e}"
            )

    def _recv(self, timeout: float = 60) -> dict:
        """Read a single JSON-RPC response line from subprocess stdout."""
        if not self._process or not self._process.stdout:
            raise MCPClientError(f"MCP server '{self._config.name}' is not running.")

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                # Process exited — collect stderr for diagnostics.
                stderr_output = ""
                if self._process.stderr:
                    try:
                        stderr_output = self._process.stderr.read()
                    except Exception:
                        pass
                self._alive = False
                raise MCPClientError(
                    f"MCP server '{self._config.name}' exited with code {self._process.returncode}. "
                    f"stderr: {stderr_output[:500]}"
                )

            try:
                line = self._process.stdout.readline()
            except Exception as e:
                self._alive = False
                raise MCPClientError(
                    f"MCP server '{self._config.name}': read failed: {e}"
                )

            if not line:
                # EOF — process probably exited between our poll check and read.
                time.sleep(0.1)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                response = json.loads(line)
            except json.JSONDecodeError as e:
                raise MCPClientError(
                    f"MCP server '{self._config.name}': invalid JSON response: {e}"
                )

            if "error" in response:
                err = response["error"]
                raise MCPClientError(
                    f"MCP server '{self._config.name}' error [{err.get('code')}]: {err.get('message')}"
                )

            return response

        raise MCPClientError(
            f"MCP server '{self._config.name}': request timed out after {timeout}s"
        )


class MCPClientProvider:
    """Implements MCPProvider protocol, wrapping multiple MCPClient instances.

    Registered in GLOBAL_EXTENSION_REGISTRY so the agent automatically
    includes external MCP tools in its tool schemas and routes calls to them.
    """

    name = "mcp_client_provider"
    description = "Provides tools from externally configured MCP servers."

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}

    def add_client(self, client: MCPClient) -> None:
        self._clients[client.name] = client

    def remove_client(self, name: str) -> None:
        if name in self._clients:
            self._clients[name].stop()
            del self._clients[name]

    def tool_schemas(self) -> list[dict]:
        """Aggregate tool schemas from all alive MCP clients."""
        schemas: list[dict] = []
        for client in self._clients.values():
            schemas.extend(client.tool_schemas())
        return schemas

    def call_tool(self, name: str, arguments: dict) -> str | None:
        """Route a tool call to the correct MCP client.

        Returns None if no client handles this tool name.
        """
        for client in self._clients.values():
            result = client.call_tool(name, arguments)
            if result is not None:
                return result
        return None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def list_servers(self) -> list[dict]:
        """Return status of all managed MCP clients."""
        result = []
        for name, client in self._clients.items():
            result.append({
                "name": name,
                "alive": client.alive,
                "tool_count": client.tool_count,
                "server_info": client._server_info,
            })
        return result

    def shutdown(self) -> None:
        """Stop all managed MCP clients."""
        for name in list(self._clients.keys()):
            self.remove_client(name)


# Singleton provider registered in the extension registry.
_MCP_PROVIDER: MCPClientProvider | None = None


def get_mcp_provider() -> MCPClientProvider:
    """Return (and create if needed) the singleton MCP client provider."""
    global _MCP_PROVIDER
    if _MCP_PROVIDER is None:
        _MCP_PROVIDER = MCPClientProvider()
    return _MCP_PROVIDER


def start_configured_mcp_servers(root: str | Path | None = None) -> MCPClientProvider:
    """Read mcp_servers.json and start all registered MCP servers.

    This is called automatically by Agent.__init__ so every agent session
    has access to configured external tools.

    Startup is best-effort: if one server fails to start, the error is
    logged and other servers still start.
    """
    provider = get_mcp_provider()

    for config in list_mcp_servers(root):
        # Skip servers that are already running.
        if config.name in provider._clients:
            continue

        try:
            client = MCPClient(config)
            client.start()
            provider.add_client(client)
        except MCPClientError as e:
            logger.warning("Failed to start MCP server '%s': %s", config.name, e)
        except Exception as e:
            logger.warning(
                "Unexpected error starting MCP server '%s': %s", config.name, e
            )

    # Register in the global extension registry so the agent picks up tools.
    if provider.client_count > 0:
        try:
            GLOBAL_EXTENSION_REGISTRY.register_mcp(provider)
        except Exception:
            pass  # already registered

    return provider
