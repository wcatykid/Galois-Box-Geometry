"""
Local web server for training and inferencing the offline Galois NLP model.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from src.webui_service import (
    REPO_ROOT,
    get_latest_model_info,
    infer_custom_pair,
    infer_synthetic_dataset,
    list_sentence_catalog,
    train_synthetic_model,
)


STATIC_DIR = REPO_ROOT / "webui" / "static"


class WebUIHandler(BaseHTTPRequestHandler):
    server_version = "GaloisWebUI/1.0"

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path, content_type: str) -> None:
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _handle_api_get(self, path: str) -> None:
        if path == "/api/status":
            self._send_json(get_latest_model_info())
            return
        if path == "/api/sentences":
            self._send_json({"sentences": list_sentence_catalog()})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str) -> None:
        body = self._read_json_body()
        if path == "/api/train":
            self._send_json(train_synthetic_model(body))
            return
        if path == "/api/infer/synthetic":
            result = infer_synthetic_dataset(
                model_path=body.get("model_path"),
                n_samples=body.get("n_samples", 120),
                seed=body.get("seed", 123),
                noise=body.get("noise", 0.05),
            )
            self._send_json(result)
            return
        if path == "/api/infer/custom":
            result = infer_custom_pair(
                sentence_1_id=body["sentence_1_id"],
                sentence_2_id=body["sentence_2_id"],
                model_path=body.get("model_path"),
            )
            self._send_json(result)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            try:
                self._handle_api_get(parsed.path)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self._handle_api_post(parsed.path)
        except KeyError as exc:
            self._send_json({"error": f"Missing field: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:
        print(f"[webui] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Galois NLP web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WebUIHandler)
    print(f"Galois NLP web UI running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
