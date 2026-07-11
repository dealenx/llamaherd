import logging
import time
from typing import Optional

from .db import db_connect


log = logging.getLogger("llamaherd")


class KeyRegistry:
    """Stores upstream Ollama Cloud keys and their cookies in the database.

    On first run (empty table), seeds from config.yaml's `keys` list.
    After that, keys added/removed/edited via the admin API persist to DB
    and survive restarts — no need to edit config.yaml manually.

    Cookies (for usage scraping) are stored alongside each key so they
    survive restarts too.
    """

    def __init__(self, db_path: str, seed_keys: list[dict] = None,
                 auth_token: Optional[str] = None, auth_user: Optional[str] = None):
        self._db_path = db_path
        self._conn = db_connect(db_path, auth_token=auth_token, auth_user=auth_user)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS upstream_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                max_concurrent INTEGER DEFAULT 15,
                cycle_day INTEGER DEFAULT 1,
                secure_session TEXT DEFAULT '',
                aid TEXT DEFAULT '',
                cf_clearance TEXT DEFAULT '',
                stripe_mid TEXT DEFAULT '',
                created REAL NOT NULL
            )
        """)
        self._conn.commit()

        # Seed from config only if table is empty
        if seed_keys and self._count() == 0:
            for kc in seed_keys:
                cookies = kc.get("cookies", {})
                self._insert(
                    token=kc["token"],
                    label=kc.get("label", f"Sub {self._count() + 1}"),
                    max_concurrent=kc.get("max_concurrent", 15),
                    cycle_day=kc.get("cycle_day", 1),
                    secure_session=cookies.get("secure_session", ""),
                    aid=cookies.get("aid", ""),
                    cf_clearance=cookies.get("cf_clearance", ""),
                    stripe_mid=cookies.get("stripe_mid", ""),
                )
            log.info(f"Seeded {len(seed_keys)} upstream keys from config to DB")

    def _count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM upstream_keys").fetchone()[0]

    def _insert(self, token: str, label: str, max_concurrent: int = 15,
                cycle_day: int = 1, secure_session: str = "", aid: str = "",
                cf_clearance: str = "", stripe_mid: str = "") -> dict:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO upstream_keys (token, label, max_concurrent, cycle_day, secure_session, aid, cf_clearance, stripe_mid, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (token, label, max_concurrent, cycle_day, secure_session, aid, cf_clearance, stripe_mid, now),
        )
        self._conn.commit()
        return {"token": token, "label": label, "max_concurrent": max_concurrent,
                "cycle_day": cycle_day, "cookies": {"secure_session": secure_session, "aid": aid,
                "cf_clearance": cf_clearance, "stripe_mid": stripe_mid}}

    def all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT token, label, max_concurrent, cycle_day, secure_session, aid, cf_clearance, stripe_mid FROM upstream_keys ORDER BY id"
        ).fetchall()
        return [{"token": r[0], "label": r[1], "max_concurrent": r[2], "cycle_day": r[3],
                 "cookies": {"secure_session": r[4] or "", "aid": r[5] or "",
                             "cf_clearance": r[6] or "", "stripe_mid": r[7] or ""}}
                for r in rows]

    def add(self, token: str, label: str, max_concurrent: int = 15, cycle_day: int = 1,
            cookies: dict = None) -> dict:
        cookies = cookies or {}
        return self._insert(token, label, max_concurrent, cycle_day,
                            secure_session=cookies.get("secure_session", ""),
                            aid=cookies.get("aid", ""),
                            cf_clearance=cookies.get("cf_clearance", ""),
                            stripe_mid=cookies.get("stripe_mid", ""))

    def update(self, token: str, label: str = None, max_concurrent: int = None,
               cycle_day: int = None) -> Optional[dict]:
        sets, params = [], []
        if label is not None:
            sets.append("label = ?")
            params.append(label)
        if max_concurrent is not None:
            sets.append("max_concurrent = ?")
            params.append(max_concurrent)
        if cycle_day is not None:
            sets.append("cycle_day = ?")
            params.append(cycle_day)
        if not sets:
            return None
        params.append(token)
        self._conn.execute(f"UPDATE upstream_keys SET {', '.join(sets)} WHERE token = ?", params)
        self._conn.commit()
        return self.get_by_token(token)

    def update_cookies(self, token: str, cookies: dict) -> Optional[dict]:
        sets, params = [], []
        field_map = {"secure_session": "secure_session", "aid": "aid",
                     "cf_clearance": "cf_clearance", "stripe_mid": "stripe_mid"}
        for k, v in cookies.items():
            if k in field_map:
                sets.append(f"{field_map[k]} = ?")
                params.append(v)
        if not sets:
            return None
        params.append(token)
        self._conn.execute(f"UPDATE upstream_keys SET {', '.join(sets)} WHERE token = ?", params)
        self._conn.commit()
        return self.get_by_token(token)

    def remove(self, token: str) -> bool:
        cur = self._conn.execute("DELETE FROM upstream_keys WHERE token = ?", [token])
        self._conn.commit()
        return cur.rowcount > 0

    def get_by_token(self, token: str) -> Optional[dict]:
        r = self._conn.execute(
            "SELECT token, label, max_concurrent, cycle_day, secure_session, aid, cf_clearance, stripe_mid FROM upstream_keys WHERE token = ?",
            [token]
        ).fetchone()
        if not r:
            return None
        return {"token": r[0], "label": r[1], "max_concurrent": r[2], "cycle_day": r[3],
                "cookies": {"secure_session": r[4] or "", "aid": r[5] or "",
                             "cf_clearance": r[6] or "", "stripe_mid": r[7] or ""}}
