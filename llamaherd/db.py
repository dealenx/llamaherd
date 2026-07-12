import logging
import base64
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

import httpx


log = logging.getLogger("llamaherd")

try:
    import libsql as _libsql  # type: ignore
    _LIBSQL_AVAILABLE = True
except ImportError:
    _libsql = None  # type: ignore
    _LIBSQL_AVAILABLE = False


def _is_libsql_url(dsn: str) -> bool:
    """True if dsn points at a remote libSQL endpoint."""
    if not dsn:
        return False
    return dsn.startswith(("http://", "https://", "libsql://", "wss://"))


def _looks_like_basic_auth(token: str) -> bool:
    """True if token looks like 'user:password' for Basic auth."""
    if not token:
        return False
    # Bearer tokens (Turso) are typically JWTs or hex strings without a colon
    # Basic auth credentials are 'user:pass' — single colon, not a URL scheme
    if ":" not in token:
        return False
    # Exclude http(s):// which would contain colons but be a URL
    if "://" in token:
        return False
    return True


class _LibSQLHTTPCursor:
    """Minimal DB-API 2.0 cursor over the libSQL /v2/pipeline HTTP API."""

    def __init__(self, conn: "_LibSQLHTTPConnection"):
        self._conn = conn
        self._rows: list[tuple] = []
        self._cols: list[str] = []
        self._pos = 0
        self.rowcount = -1
        self.lastrowid = None
        self.description: Optional[list] = None
        self.arraysize = 1

    def _convert_value(self, v: dict):
        if v is None:
            return None
        t = v.get("type")
        val = v.get("value")
        if t == "null":
            return None
        if t == "integer":
            return int(val) if val is not None else None
        if t == "float":
            return float(val) if val is not None else None
        if t == "text":
            return val
        if t == "blob":
            # Hrana returns blob values as base64-encoded strings when the request
            # used base64=True. Decode to recover the original bytes.
            return base64.b64decode(val) if isinstance(val, str) else val
        return val

    def _build_arg(self, p):
        if p is None:
            return {"type": "null"}
        if isinstance(p, bool):
            return {"type": "integer", "value": 1 if p else 0}
        if isinstance(p, int):
            return {"type": "integer", "value": p}
        if isinstance(p, float):
            return {"type": "float", "value": p}
        if isinstance(p, bytes):
            # Hrana expects base64-encoded bytes when base64=True is set.
            # Sending .hex() corrupts blob values (Reviewer #2 on PR #2).
            return {"type": "blob", "base64": True, "value": base64.b64encode(p).decode("ascii")}
        return {"type": "text", "value": str(p)}

    def _run_pipeline(self, stmt: dict):
        body = {"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]}
        resp = self._conn._client.post(
            self._conn._pipeline_url,
            headers=self._conn._headers,
            json=body,
            timeout=self._conn._timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"libSQL HTTP error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        results = data.get("results", [])
        if not results:
            raise RuntimeError("libSQL pipeline returned no results")
        first = results[0]
        if first.get("type") == "error":
            err = first.get("error", {})
            raise RuntimeError(f"libSQL error: {err.get('message', err)}")
        return first.get("response", {}).get("result", {})

    def execute(self, sql: str, params=None):
        stmt: dict = {"sql": sql}
        if params is not None:
            if isinstance(params, (list, tuple)):
                stmt["args"] = [self._build_arg(p) for p in params]
            elif isinstance(params, dict):
                stmt["named_args"] = [
                    {"name": k, "value": self._build_arg(v)} for k, v in params.items()
                ]
        result = self._run_pipeline(stmt)
        self._cols = [c.get("name", "") for c in result.get("cols", [])]
        self._rows = [
            tuple(self._convert_value(v) for v in row)
            for row in result.get("rows", [])
        ]
        self._pos = 0
        self.rowcount = result.get("affected_row_count", -1)
        self.lastrowid = result.get("last_insert_rowid")
        if self._cols:
            self.description = [(c, None, None, None, None, None, None) for c in self._cols]
        else:
            self.description = None
        return self

    def executemany(self, sql: str, seq_of_params):
        for params in seq_of_params:
            self.execute(sql, params)

    def executescript(self, sql: str):
        # /v2/pipeline supports batching — send each statement separately for simplicity
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            self.execute(stmt)

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self):
        rows = self._rows[self._pos:]
        self._pos = len(self._rows)
        return rows

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        end = min(self._pos + size, len(self._rows))
        rows = self._rows[self._pos:end]
        self._pos = end
        return rows

    def close(self):
        self._rows = []
        self._pos = 0

    def __iter__(self):
        return iter(self.fetchall())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _LibSQLHTTPConnection:
    """Minimal DB-API 2.0 connection over the libSQL /v2/pipeline HTTP API.

    Used when the native `libsql` package is unavailable (no Windows wheels)
    or when the server uses non-standard Basic auth instead of Bearer tokens.
    Implements the same execute()/commit()/cursor() surface as sqlite3.Connection
    so LlamaHerd's UsageDB / ClientRegistry work unchanged.
    """

    def __init__(self, url: str, auth_token: Optional[str] = None,
                 basic_user: Optional[str] = None, timeout: float = 30.0):
        self._pipeline_url = url.rstrip("/") + "/v2/pipeline"
        self._timeout = timeout
        import base64
        if basic_user and auth_token:
            creds = base64.b64encode(f"{basic_user}:{auth_token}".encode()).decode()
            self._headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}
        elif auth_token:
            self._headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
        else:
            self._headers = {"Content-Type": "application/json"}
        self._client = httpx.Client(timeout=timeout)

    def execute(self, sql: str, params=None):
        cur = self.cursor()
        return cur.execute(sql, params)

    def executemany(self, sql: str, seq_of_params):
        cur = self.cursor()
        return cur.executemany(sql, seq_of_params)

    def executescript(self, sql: str):
        cur = self.cursor()
        return cur.executescript(sql)

    def cursor(self):
        return _LibSQLHTTPCursor(self)

    def commit(self):
        pass  # pipeline is stateless; each request is its own transaction

    def rollback(self):
        log.warning("libSQL HTTP connection does not support rollback")

    def close(self):
        self._client.close()


def db_connect(dsn: str, auth_token: Optional[str] = None,
               auth_user: Optional[str] = None):
    """Open a DB connection. Dispatches between sqlite3, libsql, and HTTP fallback.

    - File path / :memory: → stdlib sqlite3.connect()
    - http(s):// / libsql:// with Bearer token → libsql.connect(dsn, auth_token=...)
      (requires `pip install 'llamaherd[libsql]'`)
    - http(s):// / libsql:// with Basic auth (auth_user set or token contains ':')
      → built-in _LibSQLHTTPConnection (works on any platform with httpx)
    """
    if _is_libsql_url(dsn):
        # If auth_user is explicitly set, or the token looks like "user:pass",
        # use the built-in HTTP client with Basic auth. This handles self-hosted
        # sqld deployments that reject Bearer tokens.
        use_basic = bool(auth_user) or _looks_like_basic_auth(auth_token or "")
        if use_basic:
            user = auth_user
            password = auth_token
            if not user and auth_token and ":" in auth_token:
                user, password = auth_token.split(":", 1)
            host_display = dsn.split("@")[-1] if "@" in dsn else dsn
            log.info(f"Connecting to libSQL (HTTP/Basic) at {host_display} as user '{user}'")
            return _LibSQLHTTPConnection(dsn, auth_token=password, basic_user=user)
        # Standard Turso-style Bearer auth — prefer native libsql package if available
        if _LIBSQL_AVAILABLE:
            host_display = dsn.split("@")[-1] if "@" in dsn else dsn
            log.info(f"Connecting to libSQL (native) at {host_display}")
            return _libsql.connect(dsn, auth_token=auth_token or "")
        # Fall back to HTTP client with Bearer auth
        host_display = dsn.split("@")[-1] if "@" in dsn else dsn
        log.info(f"Connecting to libSQL (HTTP/Bearer) at {host_display}")
        return _LibSQLHTTPConnection(dsn, auth_token=auth_token)
    # Local SQLite file
    path = Path(dsn).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def _safe_alter_add_column(conn, table: str, columns: list[tuple[str, str]],
                           default: Optional[str] = None) -> None:
    """Add columns to a table if they don't already exist.

    Works with both stdlib sqlite3 (raises sqlite3.OperationalError) and the
    libSQL HTTP client (raises RuntimeError with 'duplicate column' message).
    Inspects PRAGMA table_info to avoid the error entirely when possible.
    """
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {r[1] for r in rows} if rows else set()
    except Exception:
        existing = set()
    for col, typ in columns:
        if col in existing:
            continue
        default_clause = f" DEFAULT {default}" if default is not None else ""
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}{default_clause}")
        except (sqlite3.OperationalError, RuntimeError) as e:
            if "duplicate" in str(e).lower() or "already exists" in str(e).lower():
                continue
            raise


# ---------------------------------------------------------------------------
# Client Identity Registry (DB-backed, dynamic)
# ---------------------------------------------------------------------------

class ClientRegistry:
    """Maps consumer API keys to client identity for usage attribution.
    
    Backed by SQLite so keys survive restarts. Config.yaml seeds are only
    inserted on first run (if the DB is empty).
    """

    def __init__(self, db_path: str, seed_clients=None, auth_token: Optional[str] = None,
                 auth_user: Optional[str] = None):
        self._db_path = db_path
        self._conn = db_connect(db_path, auth_token=auth_token, auth_user=auth_user)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created REAL NOT NULL,
                notes TEXT DEFAULT '',
                daily_token_limit INTEGER DEFAULT NULL,
                daily_request_limit INTEGER DEFAULT NULL,
                rpm_limit INTEGER DEFAULT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_clients_token ON clients(token)")
        # Safe migration: add rate limit columns if they don't exist (existing DBs)
        # Works with both sqlite3 (raises OperationalError) and libSQL HTTP (raises RuntimeError)
        _safe_alter_add_column(self._conn, "clients", [
            ("daily_token_limit", "INTEGER"),
            ("daily_request_limit", "INTEGER"),
            ("rpm_limit", "INTEGER"),
        ], default="NULL")
        self._conn.commit()

        # In-memory cache — initialize BEFORE any _insert/_reload calls
        self._by_token: dict[str, dict] = {}
        self._by_id: dict[str, dict] = {}

        # Seed from config only if table is empty
        if seed_clients and self._count() == 0:
            for c in seed_clients:
                self._insert(c["id"], c.get("label", c["id"]), c["token"],
                             notes=c.get("notes", "seeded from config"),
                             daily_token_limit=c.get("daily_token_limit"),
                             daily_request_limit=c.get("daily_request_limit"),
                             rpm_limit=c.get("rpm_limit"))
            log.info(f"Seeded {len(seed_clients)} clients from config")
        else:
            self._reload()

    def _count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM clients").fetchone()[0]

    def _reload(self):
        """Refresh in-memory cache from DB."""
        self._by_token.clear()
        self._by_id.clear()
        rows = self._conn.execute("SELECT id, label, token, created, notes, daily_token_limit, daily_request_limit, rpm_limit FROM clients").fetchall()
        for r in rows:
            entry = {"id": r[0], "label": r[1], "token": r[2], "created": r[3], "notes": r[4],
                     "daily_token_limit": r[5], "daily_request_limit": r[6], "rpm_limit": r[7]}
            self._by_token[r[2]] = entry
            self._by_id[r[0]] = entry

    def _insert(self, client_id: str, label: str, token: str, notes: str = "",
                daily_token_limit: int = None, daily_request_limit: int = None,
                rpm_limit: int = None) -> dict:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO clients (id, label, token, created, notes, daily_token_limit, daily_request_limit, rpm_limit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (client_id, label, token, now, notes, daily_token_limit, daily_request_limit, rpm_limit),
        )
        self._conn.commit()
        self._reload()
        return {"id": client_id, "label": label, "token": token, "created": now, "notes": notes,
                "daily_token_limit": daily_token_limit, "daily_request_limit": daily_request_limit,
                "rpm_limit": rpm_limit}

    def resolve(self, token: str) -> Optional[dict]:
        """Resolve a Bearer token to a registered client identity.

        Unknown tokens are rejected by the caller. LlamaHerd used to allow
        unknown tokens through as client_id="unknown" for attribution only,
        but production deployments are API-key-only: downstream clients must
        use a token registered in the client DB.
        """
        if token in self._by_token:
            return self._by_token[token]
        # Check if it's an ID used as a token (convenience)
        if token in self._by_id:
            return self._by_id[token]
        return None

    def create(self, client_id: str, label: str, notes: str = "",
               token: Optional[str] = None,
               daily_token_limit: int = None, daily_request_limit: int = None,
               rpm_limit: int = None) -> dict:
        """Create a new client. Auto-generates a token if not provided."""
        if client_id in self._by_id:
            raise ValueError(f"client id '{client_id}' already exists")
        if not token:
            token = f"ocp-{client_id}-{secrets.token_hex(8)}"
        if token in self._by_token:
            raise ValueError(f"token already in use by client '{self._by_token[token]['id']}'")
        return self._insert(client_id, label, token, notes=notes,
                            daily_token_limit=daily_token_limit,
                            daily_request_limit=daily_request_limit,
                            rpm_limit=rpm_limit)

    def update(self, client_id: str, label: Optional[str] = None,
               notes: Optional[str] = None, token: Optional[str] = None,
               daily_token_limit: Optional[int] = ...,
               daily_request_limit: Optional[int] = ...,
               rpm_limit: Optional[int] = ...) -> Optional[dict]:
        """Update an existing client's label, notes, token, or rate limits.
        Use ... (Ellipsis) as sentinel to distinguish None (clear limit) from 'not provided'."""
        if client_id not in self._by_id:
            return None
        existing = self._by_id[client_id]
        new_label = label if label is not None else existing["label"]
        new_notes = notes if notes is not None else existing["notes"]
        new_token = token if token is not None else existing["token"]
        new_dtl = daily_token_limit if daily_token_limit is not ... else existing.get("daily_token_limit")
        new_drl = daily_request_limit if daily_request_limit is not ... else existing.get("daily_request_limit")
        new_rpm = rpm_limit if rpm_limit is not ... else existing.get("rpm_limit")
        if new_token != existing["token"] and new_token in self._by_token:
            raise ValueError(f"token already in use by client '{self._by_token[new_token]['id']}'")
        self._conn.execute(
            "UPDATE clients SET label=?, notes=?, token=?, daily_token_limit=?, daily_request_limit=?, rpm_limit=? WHERE id=?",
            (new_label, new_notes, new_token, new_dtl, new_drl, new_rpm, client_id),
        )
        self._conn.commit()
        self._reload()
        return self._by_id.get(client_id)

    def delete(self, client_id: str) -> bool:
        """Delete a client by id. Returns True if deleted."""
        cur = self._conn.execute("DELETE FROM clients WHERE id=?", (client_id,))
        self._conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            self._reload()
        return deleted

    def regenerate_token(self, client_id: str) -> dict | None:
        """Generate a new token for a client. Returns updated client or None."""
        if client_id not in self._by_id:
            return None
        new_token = f"ocp-{client_id}-{secrets.token_hex(8)}"
        self._conn.execute("UPDATE clients SET token=? WHERE id=?", (new_token, client_id))
        self._conn.commit()
        self._reload()
        return self._by_id.get(client_id)

    @property
    def clients(self) -> list[dict]:
        return list(self._by_id.values())

# ---------------------------------------------------------------------------
# Key State (upstream Ollama Cloud subscriptions)
# ---------------------------------------------------------------------------
