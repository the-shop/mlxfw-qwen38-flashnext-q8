#!/usr/bin/env python3
"""Stands in for llama-server: accepts its arg list, ignores it, serves /health
and /v1/chat/completions. Lets the gate exercise its real spawn + health path."""
import sys, json, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port = 8097; delay = 0.3
a = sys.argv[1:]
for i, v in enumerate(a):
    if v == '--port' and i + 1 < len(a): port = int(a[i+1])
print("stub llama-server starting", flush=True)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Length','15'); self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        time.sleep(delay)
        n = int(self.headers.get('content-length', 0)); self.rfile.read(n)
        b = json.dumps({"choices":[{"finish_reason":"stop","index":0,
             "message":{"role":"assistant","content":"stub answer"}}],
             "usage":{"completion_tokens":2},"timings":{"predicted_per_second":1.0}}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
print("listening on http://127.0.0.1:%d" % port, flush=True)
HTTPServer(('127.0.0.1', port), H).serve_forever()
