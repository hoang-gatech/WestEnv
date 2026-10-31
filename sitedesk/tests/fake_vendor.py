"""A tiny local HTTP server that plays a vendor's API.

The provider tests point the real vendor SDKs at it, so they exercise the SDKs' own request building
and response parsing and can assert exactly what would go over the wire, without a key or network.
"""

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeVendor:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []  # path, lower-cased headers, decoded JSON body
        self._replies: deque[tuple[int, dict[str, Any]]] = deque()
        vendor = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - the name http.server looks for
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                vendor.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "json": json.loads(body or b"{}"),
                    }
                )
                status, payload = (
                    vendor._replies.popleft()
                    if vendor._replies
                    else (500, {"error": "nothing queued"})
                )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        # a short poll interval makes shutdown() return at once instead of after 0.5 s
        threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

    def reply(self, payload: dict[str, Any], status: int = 200) -> None:
        self._replies.append((status, payload))

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
