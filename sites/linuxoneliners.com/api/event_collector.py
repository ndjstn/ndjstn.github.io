#!/usr/bin/env python3
"""First-party event collector for linuxoneliners.com."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_BODY_BYTES = 32_768
RATE_WINDOW_SECONDS = 10
RATE_LIMIT_PER_WINDOW = 120

REQUEST_COUNTS: dict[str, list[float]] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            create table if not exists events (
              id integer primary key autoincrement,
              received_at_ms integer not null,
              event_name text not null,
              site text not null,
              session_id text,
              visitor_id text,
              path text,
              page_title text,
              referrer text,
              label text,
              properties_json text,
              performance_json text,
              viewport_json text,
              connection_json text,
              language text,
              timezone text,
              user_agent text,
              ip_address text,
              method text,
              content_length integer,
              ingest_ms integer not null
            )
            """
        )
        conn.execute("create index if not exists idx_events_received on events(received_at_ms)")
        conn.execute("create index if not exists idx_events_name on events(event_name)")
        conn.execute("create index if not exists idx_events_path on events(path)")
        conn.execute("create index if not exists idx_events_session on events(session_id)")


def compact_json(value: Any) -> str:
    return json.dumps(value if isinstance(value, (dict, list)) else {}, separators=(",", ":"))


def client_ip(headers: Any, fallback: str) -> str:
    forwarded_for = headers.get("X-Forwarded-For", "")
    if forwarded_for:
      return forwarded_for.split(",", 1)[0].strip()
    real_ip = headers.get("X-Real-IP", "")
    return real_ip.strip() or fallback


def rate_limited(ip: str) -> bool:
    cutoff = time.time() - RATE_WINDOW_SECONDS
    recent = [ts for ts in REQUEST_COUNTS.get(ip, []) if ts >= cutoff]
    if len(recent) >= RATE_LIMIT_PER_WINDOW:
        REQUEST_COUNTS[ip] = recent
        return True
    recent.append(time.time())
    REQUEST_COUNTS[ip] = recent
    return False


class EventHandler(BaseHTTPRequestHandler):
    server_version = "LinuxOneLinersAnalytics/0.1"

    def do_GET(self) -> None:
        if self.path != "/api/health":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self) -> None:
        started = now_ms()
        if self.path != "/api/events":
            self.send_error(404)
            return

        ip = client_ip(self.headers, self.client_address[0])
        if rate_limited(ip):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"rate_limited"}')
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"bad_size"}')
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"bad_json"}')
            return

        event_name = str(payload.get("name", "unknown"))[:120]
        properties = payload.get("properties") if isinstance(payload.get("properties"), dict) else {}

        row = {
            "received_at_ms": now_ms(),
            "event_name": event_name,
            "site": str(payload.get("site", "linuxoneliners.com"))[:120],
            "session_id": str(payload.get("sessionId", ""))[:160],
            "visitor_id": str(payload.get("visitorId", ""))[:160],
            "path": str(payload.get("path", ""))[:500],
            "page_title": str(payload.get("pageTitle", ""))[:500],
            "referrer": str(payload.get("referrer", ""))[:1000],
            "label": str(properties.get("label", ""))[:240],
            "properties_json": compact_json(properties),
            "performance_json": compact_json(payload.get("performance")),
            "viewport_json": compact_json(payload.get("viewport")),
            "connection_json": compact_json(payload.get("connection")),
            "language": str(payload.get("language", ""))[:80],
            "timezone": str(payload.get("timezone", ""))[:120],
            "user_agent": str(self.headers.get("User-Agent", ""))[:1000],
            "ip_address": ip[:120],
            "method": self.command,
            "content_length": content_length,
            "ingest_ms": max(0, now_ms() - started),
        }

        with sqlite3.connect(self.server.db_path) as conn:  # type: ignore[attr-defined]
            conn.execute(
                """
                insert into events (
                  received_at_ms, event_name, site, session_id, visitor_id, path,
                  page_title, referrer, label, properties_json, performance_json,
                  viewport_json, connection_json, language, timezone, user_agent,
                  ip_address, method, content_length, ingest_ms
                ) values (
                  :received_at_ms, :event_name, :site, :session_id, :visitor_id, :path,
                  :page_title, :referrer, :label, :properties_json, :performance_json,
                  :viewport_json, :connection_json, :language, :timezone, :user_agent,
                  :ip_address, :method, :content_length, :ingest_ms
                )
                """,
                row,
            )

        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("LOL_ANALYTICS_ACCESS_LOG", "0") == "1":
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("LOL_ANALYTICS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOL_ANALYTICS_PORT", "8091")))
    parser.add_argument(
        "--db",
        default=os.environ.get("LOL_ANALYTICS_DB", "/srv/data/linuxoneliners.com/analytics.sqlite3"),
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    init_db(db_path)

    server = ThreadingHTTPServer((args.host, args.port), EventHandler)
    server.db_path = str(db_path)  # type: ignore[attr-defined]
    print(f"analytics listening on http://{args.host}:{args.port}, db={db_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
