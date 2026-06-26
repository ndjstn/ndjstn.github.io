#!/usr/bin/env python3
"""First-party Google SSO comments for justinstone.online."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_BODY_BYTES = 16_384
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def now_ms() -> int:
    return int(time.time() * 1000)


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            create table if not exists users (
              id integer primary key autoincrement,
              provider text not null,
              provider_subject text not null,
              email text,
              email_verified integer not null default 0,
              display_name text,
              picture_url text,
              created_at_ms integer not null,
              last_seen_at_ms integer not null,
              unique(provider, provider_subject)
            )
            """
        )
        conn.execute(
            """
            create table if not exists sessions (
              id text primary key,
              user_id integer not null references users(id) on delete cascade,
              created_at_ms integer not null,
              expires_at_ms integer not null,
              ip_address text,
              user_agent text
            )
            """
        )
        conn.execute(
            """
            create table if not exists comments (
              id integer primary key autoincrement,
              page_url text not null,
              page_title text,
              user_id integer not null references users(id) on delete cascade,
              body text not null,
              status text not null default 'pending',
              created_at_ms integer not null,
              ip_address text,
              user_agent text
            )
            """
        )
        conn.execute("create index if not exists idx_comments_page_status on comments(page_url, status)")
        conn.execute("create index if not exists idx_comments_status on comments(status)")
        conn.execute("create index if not exists idx_sessions_expires on sessions(expires_at_ms)")


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def client_ip(headers: Any, fallback: str) -> str:
    forwarded_for = headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    real_ip = headers.get("X-Real-IP", "")
    return real_ip.strip() or fallback


def parse_cookie(header: str | None) -> cookies.SimpleCookie[str]:
    jar: cookies.SimpleCookie[str] = cookies.SimpleCookie()
    if header:
        jar.load(header)
    return jar


def cookie_header(name: str, value: str, max_age: int = SESSION_TTL_SECONDS) -> str:
    morsel = cookies.SimpleCookie()
    morsel[name] = value
    morsel[name]["path"] = "/"
    morsel[name]["max-age"] = str(max_age)
    morsel[name]["httponly"] = True
    morsel[name]["secure"] = True
    morsel[name]["samesite"] = "Lax"
    return morsel.output(header="").strip()


def clear_cookie_header(name: str) -> str:
    return f"{name}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax"


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def sign_value(secret: str, value: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    return f"{value}.{b64url(digest)}"


def verify_signed_value(secret: str, signed: str) -> str | None:
    if "." not in signed:
        return None
    value, signature = signed.rsplit(".", 1)
    expected = sign_value(secret, value).rsplit(".", 1)[1]
    if hmac.compare_digest(signature, expected):
        return value
    return None


def request_json(url: str, data: dict[str, str] | None = None) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(url, data=encoded, method="POST" if data is not None else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def configured(server: Any) -> bool:
    return bool(server.google_client_id and server.google_client_secret and server.public_base_url)


class CommentHandler(BaseHTTPRequestHandler):
    server_version = "JustinStoneComments/0.1"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/comments/health":
            json_response(self, 200, {"ok": True, "configured": configured(self.server)})  # type: ignore[attr-defined]
            return

        if parsed.path == "/api/me":
            user = self.current_user()
            json_response(self, 200, {"authenticated": bool(user), "user": user})
            return

        if parsed.path == "/api/comments":
            self.list_comments(parsed)
            return

        if parsed.path == "/auth/google/start":
            self.google_start(parsed)
            return

        if parsed.path == "/auth/google/callback":
            self.google_callback(parsed)
            return

        if parsed.path == "/auth/logout":
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", clear_cookie_header("js_session"))
            self.end_headers()
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/comments":
            self.send_error(404)
            return

        user = self.current_user()
        if not user:
            json_response(self, 401, {"ok": False, "error": "auth_required"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            json_response(self, 413, {"ok": False, "error": "bad_size"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            json_response(self, 400, {"ok": False, "error": "bad_json"})
            return

        if str(payload.get("website", "")).strip():
            json_response(self, 202, {"ok": True, "status": "pending"})
            return

        page_url = str(payload.get("page_url", ""))[:500]
        page_title = str(payload.get("page_title", ""))[:500]
        body = str(payload.get("body", "")).strip()

        if not page_url.startswith("/") or len(body) < 3:
            json_response(self, 400, {"ok": False, "error": "bad_comment"})
            return

        with sqlite3.connect(self.server.db_path) as conn:  # type: ignore[attr-defined]
            conn.execute(
                """
                insert into comments (
                  page_url, page_title, user_id, body, status, created_at_ms, ip_address, user_agent
                ) values (?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    page_url,
                    page_title,
                    user["id"],
                    body[:4000],
                    now_ms(),
                    client_ip(self.headers, self.client_address[0])[:120],
                    str(self.headers.get("User-Agent", ""))[:1000],
                ),
            )

        json_response(self, 202, {"ok": True, "status": "pending"})

    def list_comments(self, parsed: urllib.parse.ParseResult) -> None:
        query = urllib.parse.parse_qs(parsed.query)
        page_url = query.get("page_url", [""])[0][:500]
        if not page_url.startswith("/"):
            json_response(self, 400, {"ok": False, "error": "bad_page_url"})
            return

        with sqlite3.connect(self.server.db_path) as conn:  # type: ignore[attr-defined]
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select comments.id, comments.body, comments.created_at_ms, users.display_name
                from comments
                join users on users.id = comments.user_id
                where comments.page_url = ? and comments.status = 'approved'
                order by comments.created_at_ms asc
                limit 100
                """,
                (page_url,),
            ).fetchall()

        comments_payload = [
            {
                "id": row["id"],
                "body": row["body"],
                "author": row["display_name"] or "Reader",
                "created_at": datetime.fromtimestamp(row["created_at_ms"] / 1000, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
            for row in rows
        ]
        json_response(self, 200, {"ok": True, "comments": comments_payload})

    def google_start(self, parsed: urllib.parse.ParseResult) -> None:
        if not configured(self.server):  # type: ignore[attr-defined]
            json_response(self, 503, {"ok": False, "error": "sso_not_configured"})
            return

        query = urllib.parse.parse_qs(parsed.query)
        next_path = query.get("next", ["/"])[0]
        if not next_path.startswith("/"):
            next_path = "/"

        nonce = secrets.token_urlsafe(24)
        state = sign_value(self.server.cookie_secret, json.dumps({"nonce": nonce, "next": next_path}))  # type: ignore[attr-defined]
        params = {
            "client_id": self.server.google_client_id,  # type: ignore[attr-defined]
            "redirect_uri": f"{self.server.public_base_url}/auth/google/callback",  # type: ignore[attr-defined]
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "prompt": "select_account",
        }
        self.send_response(302)
        self.send_header("Location", f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}")
        self.send_header("Set-Cookie", cookie_header("js_oauth_state", state, max_age=600))
        self.end_headers()

    def google_callback(self, parsed: urllib.parse.ParseResult) -> None:
        query = urllib.parse.parse_qs(parsed.query)
        code = query.get("code", [""])[0]
        state = query.get("state", [""])[0]
        jar = parse_cookie(self.headers.get("Cookie"))
        cookie_state = jar.get("js_oauth_state")

        if not code or not state or not cookie_state or cookie_state.value != state:
            self.send_error(400, "bad oauth state")
            return

        state_value = verify_signed_value(self.server.cookie_secret, state)  # type: ignore[attr-defined]
        if not state_value:
            self.send_error(400, "bad oauth state")
            return

        state_payload = json.loads(state_value)
        next_path = state_payload.get("next", "/")
        if not isinstance(next_path, str) or not next_path.startswith("/"):
            next_path = "/"

        token = request_json(
            GOOGLE_TOKEN_URL,
            {
                "code": code,
                "client_id": self.server.google_client_id,  # type: ignore[attr-defined]
                "client_secret": self.server.google_client_secret,  # type: ignore[attr-defined]
                "redirect_uri": f"{self.server.public_base_url}/auth/google/callback",  # type: ignore[attr-defined]
                "grant_type": "authorization_code",
            },
        )
        id_token = token.get("id_token")
        if not isinstance(id_token, str):
            self.send_error(401, "missing id token")
            return

        profile = request_json(f"{GOOGLE_TOKENINFO_URL}?{urllib.parse.urlencode({'id_token': id_token})}")
        if profile.get("aud") != self.server.google_client_id:  # type: ignore[attr-defined]
            self.send_error(401, "bad audience")
            return
        if profile.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
            self.send_error(401, "bad issuer")
            return

        user_id = self.upsert_user(profile)
        session_id = secrets.token_urlsafe(32)
        expires_at = now_ms() + SESSION_TTL_SECONDS * 1000
        with sqlite3.connect(self.server.db_path) as conn:  # type: ignore[attr-defined]
            conn.execute(
                "insert into sessions (id, user_id, created_at_ms, expires_at_ms, ip_address, user_agent) values (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    user_id,
                    now_ms(),
                    expires_at,
                    client_ip(self.headers, self.client_address[0])[:120],
                    str(self.headers.get("User-Agent", ""))[:1000],
                ),
            )

        self.send_response(302)
        self.send_header("Location", next_path)
        self.send_header("Set-Cookie", clear_cookie_header("js_oauth_state"))
        self.send_header("Set-Cookie", cookie_header("js_session", session_id))
        self.end_headers()

    def upsert_user(self, profile: dict[str, Any]) -> int:
        provider_subject = str(profile.get("sub", ""))[:240]
        email = str(profile.get("email", ""))[:240]
        display_name = str(profile.get("name", ""))[:240]
        picture_url = str(profile.get("picture", ""))[:1000]
        email_verified = 1 if str(profile.get("email_verified", "")).lower() == "true" else 0

        if not provider_subject:
            raise ValueError("missing google subject")

        with sqlite3.connect(self.server.db_path) as conn:  # type: ignore[attr-defined]
            conn.execute(
                """
                insert into users (
                  provider, provider_subject, email, email_verified, display_name, picture_url, created_at_ms, last_seen_at_ms
                ) values ('google', ?, ?, ?, ?, ?, ?, ?)
                on conflict(provider, provider_subject) do update set
                  email = excluded.email,
                  email_verified = excluded.email_verified,
                  display_name = excluded.display_name,
                  picture_url = excluded.picture_url,
                  last_seen_at_ms = excluded.last_seen_at_ms
                """,
                (provider_subject, email, email_verified, display_name, picture_url, now_ms(), now_ms()),
            )
            row = conn.execute(
                "select id from users where provider = 'google' and provider_subject = ?",
                (provider_subject,),
            ).fetchone()
            return int(row[0])

    def current_user(self) -> dict[str, Any] | None:
        jar = parse_cookie(self.headers.get("Cookie"))
        session = jar.get("js_session")
        if not session:
            return None

        with sqlite3.connect(self.server.db_path) as conn:  # type: ignore[attr-defined]
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                select users.id, users.email, users.display_name, users.picture_url
                from sessions
                join users on users.id = sessions.user_id
                where sessions.id = ? and sessions.expires_at_ms > ?
                """,
                (session.value, now_ms()),
            ).fetchone()

        if not row:
            return None
        return {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"],
            "picture_url": row["picture_url"],
        }

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("JS_COMMENTS_ACCESS_LOG", "0") == "1":
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("JS_COMMENTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JS_COMMENTS_PORT", "8092")))
    parser.add_argument("--db", default=os.environ.get("JS_COMMENTS_DB", "/srv/data/justinstone.online/comments.sqlite3"))
    args = parser.parse_args()

    db_path = Path(args.db)
    init_db(db_path)

    server = ThreadingHTTPServer((args.host, args.port), CommentHandler)
    server.db_path = str(db_path)  # type: ignore[attr-defined]
    server.google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")  # type: ignore[attr-defined]
    server.google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")  # type: ignore[attr-defined]
    server.public_base_url = os.environ.get("JS_COMMENTS_PUBLIC_BASE_URL", "https://justinstone.online")  # type: ignore[attr-defined]
    server.cookie_secret = os.environ.get("JS_COMMENTS_COOKIE_SECRET") or secrets.token_urlsafe(48)  # type: ignore[attr-defined]
    print(f"comments listening on http://{args.host}:{args.port}, db={db_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
