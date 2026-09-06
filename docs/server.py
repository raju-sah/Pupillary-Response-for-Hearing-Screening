"""
Lightweight zero-dependency HTTP server for the AEPR Optical Audiometry Web Dashboard.
Usage:
    python webapp/server.py [--port 8000]
"""

import http.server
import socketserver
import os
import sys
import argparse
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBAPP_DIR), **kwargs)

    def end_headers(self):
        # Disable caching for instant updates during development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def run_server(port: int = 8000):
    handler = CustomHTTPRequestHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print("==============================================================")
        print("  AEPR OPTICAL AUDIOMETRY CLINICAL DASHBOARD")
        print(f"  Local URL: http://localhost:{port}")
        print("  Press Ctrl+C to stop the server.")
        print("==============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEPR Webapp Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
