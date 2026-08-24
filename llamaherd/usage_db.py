import sqlite3
import time
from datetime import UTC, datetime

from .db import _safe_alter_add_column, db_connect


class UsageDB:
    def __init__(self, db_path: str, auth_token: str | None = None,
                 auth_user: str | None = None):
        self.db_path = db_path
        self._conn = db_connect(db_path, auth_token=auth_token, auth_user=auth_user)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                ts REAL NOT NULL,
                day TEXT NOT NULL,
                client_id TEXT NOT NULL,
                upstream_key TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                status INTEGER NOT NULL,
                session_id TEXT DEFAULT ''
            )
            """)
        # Backward-compatible migration for existing DBs
        _safe_alter_add_column(self._conn, "usage", [("session_id", "TEXT")], default="''")
        # Drop obsolete quota columns if present
        for col in ["session_usage_pct", "weekly_usage_pct"]:
            try:
                self._conn.execute(f"ALTER TABLE usage DROP COLUMN {col}")
            except (sqlite3.OperationalError, RuntimeError):
                pass
        for idx_cols in [
            "client_id, day",
            "model, day",
            "day",
            "client_id, model, day",
            "session_id",
        ]:
            idx_name = f"idx_usage_{'_'.join(idx_cols.replace(' ', '').split(','))}"
            try:
                self._conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON usage ({idx_cols})")
            except (sqlite3.OperationalError, RuntimeError):
                pass
        self._conn.commit()

    def record(self, client_id: str, upstream_key: str, model: str,
               tokens_in: int, tokens_out: int, latency_ms: int, status: int,
               session_id: str = ""):
        today = datetime.now(UTC).date().isoformat()
        self._conn.execute(
            "INSERT INTO usage (ts, day, client_id, upstream_key, model, tokens_in, tokens_out, latency_ms, status, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), today, client_id, upstream_key, model,
             tokens_in, tokens_out, latency_ms, status, session_id or ""),
        )
        self._conn.commit()

    def summary(self, hours: int = 24, client: str | None = None, model: str | None = None) -> list[dict]:
        since = time.time() - hours * 3600
        query = """
            SELECT client_id, model, day,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total,
                   AVG(latency_ms) as avg_latency_ms
            FROM usage WHERE ts > ?
        """
        params: list = [since]
        if client:
            query += " AND client_id = ?"
            params.append(client)
        if model:
            query += " AND model = ?"
            params.append(model)
        query += " GROUP BY client_id, model, day ORDER BY day DESC, tokens_total DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [{
            "client_id": r[0],
            "model": r[1],
            "day": r[2],
            "requests": r[3],
            "tokens_in": r[4] or 0,
            "tokens_out": r[5] or 0,
            "tokens_total": r[6] or 0,
            "avg_latency_ms": round(r[7] or 0, 1),
        } for r in rows]

    def _date_range_where(self, days: int | None = None, start_date: str | None = None, end_date: str | None = None):
        """Build WHERE clause + params for date range filtering.
        Supports both `days` (relative) and start_date/end_date (absolute ISO date).
        Returns (where_clause, params).
        """
        if start_date and end_date:
            return "day >= ? AND day <= ?", [start_date, end_date]
        elif start_date:
            return "day >= ?", [start_date]
        elif end_date:
            return "day <= ?", [end_date]
        else:
            since = time.time() - (days or 30) * 86400
            return "ts > ?", [since]

    def daily_totals(self, days: int = 30, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT day,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where}
            GROUP BY day ORDER BY day DESC
        """, params).fetchall()
        return [{
            "day": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in rows]

    def by_client(self, days: int = 30, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT client_id,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where}
            GROUP BY client_id ORDER BY tokens_total DESC
        """, params).fetchall()
        return [{
            "client_id": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in rows]

    def by_upstream_key(self, days: int = 30, start_date: str | None = None,
                        end_date: str | None = None) -> list[dict]:
        """Aggregate usage per upstream key (Ollama Cloud subscription).

        Groups by upstream_key (first 8 chars of token), excluding 'none'
        (error paths with no key assigned).
        """
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT upstream_key,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where} AND status != -1 AND upstream_key != 'none'
            GROUP BY upstream_key ORDER BY tokens_total DESC
        """, params).fetchall()
        return [{
            "upstream_key": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in rows]

    def upstream_key_costs(self, pricing: dict, days: int = 30,
                           start_date: str | None = None, end_date: str | None = None,
                           key_labels: dict | None = None) -> dict:
        """Calculate per-key OpenRouter equivalent costs.

        Groups by upstream_key × model, then aggregates cost per key.
        Accepts key_labels dict (upstream_key -> {label, plan}) for enrichment.
        """
        where, params = self._date_range_where(days, start_date, end_date)
        rows = self._conn.execute(f"""
            SELECT upstream_key, model,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   COUNT(*) as requests
            FROM usage WHERE {where} AND status != -1 AND upstream_key != 'none'
            GROUP BY upstream_key, model
            ORDER BY upstream_key, SUM(tokens_in + tokens_out) DESC
        """, params).fetchall()

        key_labels = key_labels or {}
        keys_map: dict[str, dict] = {}
        total_cost = 0.0
        total_input_cost = 0.0
        total_output_cost = 0.0
        unpriced = []

        for r in rows:
            upstream_key = r[0]
            model_raw = r[1]
            tokens_in = r[2] or 0
            tokens_out = r[3] or 0
            requests = r[4] or 0

            lookup_key = model_raw.replace(":cloud", "").replace(":cloud-", "-")
            p = pricing.get(lookup_key) or pricing.get(model_raw)

            if p:
                in_cost = tokens_in / 1_000_000 * p.get("input_per_1m", 0)
                out_cost = tokens_out / 1_000_000 * p.get("output_per_1m", 0)
                cost = in_cost + out_cost
            else:
                in_cost = 0.0
                out_cost = 0.0
                cost = 0.0
                if model_raw not in unpriced:
                    unpriced.append(model_raw)

            total_cost += cost
            total_input_cost += in_cost
            total_output_cost += out_cost

            key_entry = keys_map.setdefault(upstream_key, {
                "upstream_key": upstream_key,
                "label": key_labels.get(upstream_key, {}).get("label", upstream_key),
                "plan": key_labels.get(upstream_key, {}).get("plan", ""),
                "account_email": key_labels.get(upstream_key, {}).get("account_email", ""),
                "requests": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "total_cost_usd": 0.0,
                "total_input_cost_usd": 0.0,
                "total_output_cost_usd": 0.0,
                "models": [],
            })
            key_entry["requests"] += requests
            key_entry["tokens_in"] += tokens_in
            key_entry["tokens_out"] += tokens_out
            key_entry["total_cost_usd"] += cost
            key_entry["total_input_cost_usd"] += in_cost
            key_entry["total_output_cost_usd"] += out_cost
            key_entry["models"].append({
                "model": model_raw,
                "openrouter_id": p.get("openrouter_id", "") if p else "",
                "requests": requests,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "input_cost_usd": round(in_cost, 4),
                "output_cost_usd": round(out_cost, 4),
                "total_cost_usd": round(cost, 4),
                "input_per_1m": p.get("input_per_1m") if p else None,
                "output_per_1m": p.get("output_per_1m") if p else None,
            })

        keys_list = list(keys_map.values())
        for k in keys_list:
            k["total_cost_usd"] = round(k["total_cost_usd"], 2)
            k["total_input_cost_usd"] = round(k["total_input_cost_usd"], 2)
            k["total_output_cost_usd"] = round(k["total_output_cost_usd"], 2)
        keys_list.sort(key=lambda x: -x["total_cost_usd"])

        return {
            "keys": keys_list,
            "total_cost_usd": round(total_cost, 2),
            "total_input_cost_usd": round(total_input_cost, 2),
            "total_output_cost_usd": round(total_output_cost, 2),
            "unpriced_models": unpriced,
        }

    def upstream_key_detail(self, key_prefix: str, days: int = 30,
                            start_date: str | None = None, end_date: str | None = None) -> dict:
        """Detailed breakdown for a single upstream key.

        Returns per-model, per-client, per-day, and per-status breakdowns.
        """
        where, params = self._date_range_where(days, start_date, end_date)
        params_with_key = params + [key_prefix]

        model_rows = self._conn.execute(f"""
            SELECT model,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total,
                   AVG(latency_ms) as avg_latency_ms
            FROM usage WHERE {where} AND upstream_key = ? AND status != -1
            GROUP BY model ORDER BY tokens_total DESC
        """, params_with_key).fetchall()

        client_rows = self._conn.execute(f"""
            SELECT client_id,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where} AND upstream_key = ? AND status != -1
            GROUP BY client_id ORDER BY tokens_total DESC
        """, params_with_key).fetchall()

        daily_rows = self._conn.execute(f"""
            SELECT day,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total
            FROM usage WHERE {where} AND upstream_key = ? AND status != -1
            GROUP BY day ORDER BY day DESC
        """, params_with_key).fetchall()

        status_rows = self._conn.execute(f"""
            SELECT status, COUNT(*) as count
            FROM usage WHERE {where} AND upstream_key = ?
            GROUP BY status
        """, params_with_key).fetchall()

        models = [{
            "model": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
            "avg_latency_ms": round(r[5] or 0, 1),
        } for r in model_rows]

        clients = [{
            "client_id": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in client_rows]

        daily = [{
            "day": r[0],
            "requests": r[1],
            "tokens_in": r[2] or 0,
            "tokens_out": r[3] or 0,
            "tokens_total": r[4] or 0,
        } for r in daily_rows]

        status_counts = [{"status": r[0], "count": r[1]} for r in status_rows]

        total_requests = sum(m["requests"] for m in models)
        total_tokens_in = sum(m["tokens_in"] for m in models)
        total_tokens_out = sum(m["tokens_out"] for m in models)

        return {
            "upstream_key": key_prefix,
            "models": models,
            "clients": clients,
            "daily": daily,
            "status_counts": status_counts,
            "totals": {
                "requests": total_requests,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "tokens_total": total_tokens_in + total_tokens_out,
            },
        }

    def quota_coefficients(self, manager_keys: list, days: int = 7) -> dict:
        """Derive implied Ollama quota cost per token for each model.

        Combines the per-key weekly quota-share bars scraped from ollama.com/settings
        with actual token usage in the local DB. Models appearing in both the
        scraped quota data and the DB with non-trivial usage get a coefficient.
        """
        if not manager_keys:
            return {"models": [], "note": "No keys configured"}

        since = time.time() - days * 86400
        rows = self._conn.execute("""
            SELECT model,
                   COUNT(*) as requests,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   SUM(tokens_in + tokens_out) as tokens_total,
                   upstream_key
            FROM usage WHERE ts > ? AND status != -1
            GROUP BY upstream_key, model
        """, [since]).fetchall()

        # Aggregate totals per key and per model across all keys
        key_totals: dict[str, int] = {}
        model_key_data: dict[str, list[tuple[str, int, float]]] = {}
        for model, requests, tokens_in, tokens_out, tokens_total, upstream_key in rows:
            if tokens_total <= 0:
                continue
            key_totals[upstream_key] = key_totals.get(upstream_key, 0) + tokens_total
            model_key_data.setdefault(model, []).append(
                (upstream_key, tokens_total, requests)
            )

        # Build per-model coefficients by averaging across keys where both
        # scraped quota share and DB usage are available.
        out = []
        for model, key_list in model_key_data.items():
            ratios = []
            quota_shares = []
            token_shares = []
            calls = 0
            tokens = 0
            for upstream_key, tokens_total, requests in key_list:
                key = next((k for k in manager_keys if k.token[:8] == upstream_key), None)
                if not key or not key.weekly_models:
                    continue
                weekly = key.weekly_models.get(model, {})
                quota_pct = weekly.get("bar_pct", 0.0) or 0.0
                if quota_pct <= 0:
                    continue
                total_for_key = key_totals.get(upstream_key, 1) or 1
                token_pct = 100.0 * tokens_total / total_for_key
                if token_pct <= 0:
                    continue
                ratios.append(quota_pct / token_pct)
                quota_shares.append(quota_pct)
                token_shares.append(token_pct)
                calls += requests
                tokens += tokens_total
            if not ratios:
                continue
            avg_ratio = sum(ratios) / len(ratios)
            avg_quota = sum(quota_shares) / len(quota_shares)
            avg_token = sum(token_shares) / len(token_shares)
            out.append({
                "model": model,
                "coefficient": round(avg_ratio, 2),
                "quota_share_pct": round(avg_quota, 2),
                "token_share_pct": round(avg_token, 2),
                "requests": calls,
                "tokens": tokens,
                "keys_used": len(ratios),
            })

        if not out:
            return {"models": [], "note": f"No model had both scraped quota share and DB usage in the last {days} days"}

        # Normalize so the cheapest model (by coefficient) is 1.0
        min_coeff = min(m["coefficient"] for m in out)
        if min_coeff > 0:
            for m in out:
                m["relative"] = round(m["coefficient"] / min_coeff, 1)
        else:
            for m in out:
                m["relative"] = None

        out.sort(key=lambda x: -x["coefficient"])
        return {"models": out, "note": f"Implied Ollama quota cost per token, averaged across keys (last {days} days)."}

    def openrouter_costs(self, pricing: dict, days: int = 30,
                         start_date: str | None = None, end_date: str | None = None,
                         client_id: str | None = None, alias_manager=None) -> dict:
        """Calculate what the usage WOULD have cost on OpenRouter.

        Args:
            pricing: dict from openrouter_pricing.yaml, keyed by model name.
                     Each value has 'input_per_1m' and 'output_per_1m'.
            days/start_date/end_date: time range filter.
            client_id: optional filter by client.

        Returns dict with 'models' (per-model breakdown), 'total_cost',
        'total_input_cost', 'total_output_cost', 'unpriced_models'.
        """
        where, params = self._date_range_where(days, start_date, end_date)
        extras = [" AND status != -1"]
        if client_id:
            extras.append(" AND client_id = ?")
            params.append(client_id)
        query = f"""
            SELECT model,
                   SUM(tokens_in) as tokens_in,
                   SUM(tokens_out) as tokens_out,
                   COUNT(*) as requests
            FROM usage WHERE {where}{''.join(extras)}
            GROUP BY model ORDER BY SUM(tokens_in + tokens_out) DESC
        """
        rows = self._conn.execute(query, params).fetchall()

        models = []
        total_cost = 0.0
        total_input_cost = 0.0
        total_output_cost = 0.0
        unpriced = []

        for r in rows:
            model_raw = r[0]
            tokens_in = r[1] or 0
            tokens_out = r[2] or 0
            requests = r[3] or 0

            # Strip :cloud suffix for lookup
            lookup_key = model_raw.replace(":cloud", "").replace(":cloud-", "-")
            # Resolve model aliases (e.g. glm-5.2-256k → glm-5.2) so cost
            # attribution inherits the upstream model's pricing.
            if alias_manager and alias_manager.is_alias(lookup_key):
                upstream, _ = alias_manager.resolve(lookup_key)
                lookup_key = upstream
            p = pricing.get(lookup_key) or pricing.get(model_raw)
            # Fallback: strip -NNNk / -NNNm context suffixes from old alias
            # names (e.g. glm-5.2-256k → glm-5.2) so historical usage rows
            # still resolve to the base model's pricing after alias renames.
            if not p:
                import re as _re
                m = _re.match(r'^(.+)-(\d+[kmb])$', lookup_key, _re.IGNORECASE)
                if m:
                    p = pricing.get(m.group(1))

            if p:
                in_cost = tokens_in / 1_000_000 * p.get("input_per_1m", 0)
                out_cost = tokens_out / 1_000_000 * p.get("output_per_1m", 0)
                cost = in_cost + out_cost
                total_cost += cost
                total_input_cost += in_cost
                total_output_cost += out_cost
                models.append({
                    "model": model_raw,
                    "openrouter_id": p.get("openrouter_id", ""),
                    "requests": requests,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "input_cost_usd": round(in_cost, 4),
                    "output_cost_usd": round(out_cost, 4),
                    "total_cost_usd": round(cost, 4),
                    "input_per_1m": p.get("input_per_1m", 0),
                    "output_per_1m": p.get("output_per_1m", 0),
                })
            else:
                unpriced.append(model_raw)
                models.append({
                    "model": model_raw,
                    "openrouter_id": "",
                    "requests": requests,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "input_cost_usd": 0,
                    "output_cost_usd": 0,
                    "total_cost_usd": 0,
                    "input_per_1m": None,
                    "output_per_1m": None,
                })

        return {
            "models": models,
            "total_cost_usd": round(total_cost, 2),
            "total_input_cost_usd": round(total_input_cost, 2),
            "total_output_cost_usd": round(total_output_cost, 2),
            "unpriced_models": unpriced,
        }

    def recent_calls(self, limit: int = 100, start_date: str | None = None, end_date: str | None = None,
                     client_id: str | None = None, model: str | None = None) -> list[dict]:
        where_parts = []
        params: list = []
        if start_date:
            where_parts.append("day >= ?")
            params.append(start_date)
        if end_date:
            where_parts.append("day <= ?")
            params.append(end_date)
        if client_id:
            where_parts.append("client_id = ?")
            params.append(client_id)
        if model:
            where_parts.append("model = ?")
            params.append(model)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        query = f"""
            SELECT ts, client_id, upstream_key, model,
                   tokens_in, tokens_out, latency_ms, status
            FROM usage WHERE {where}
            ORDER BY ts DESC LIMIT ?
        """
        params.append(min(limit, 500))
        rows = self._conn.execute(query, params).fetchall()
        return [{
            "ts": r[0],
            "time": datetime.fromtimestamp(r[0], tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "client_id": r[1],
            "upstream_key": r[2],
            "model": r[3],
            "tokens_in": r[4],
            "tokens_out": r[5],
            "tokens_total": r[4] + r[5],
            "latency_ms": r[6],
            "status": r[7],
        } for r in rows]

    def totals(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        if start_date or end_date:
            where, params = self._date_range_where(None, start_date, end_date)
            row = self._conn.execute(f"""
                SELECT COUNT(*),
                       COALESCE(SUM(tokens_in), 0),
                       COALESCE(SUM(tokens_out), 0),
                       AVG(latency_ms),
                       100.0 * SUM(CASE WHEN status < 200 OR status >= 400 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
                FROM usage WHERE {where}
            """, params).fetchone()
        else:
            row = self._conn.execute("""
                SELECT COUNT(*),
                       COALESCE(SUM(tokens_in), 0),
                       COALESCE(SUM(tokens_out), 0),
                       AVG(latency_ms),
                       100.0 * SUM(CASE WHEN status < 200 OR status >= 400 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
                FROM usage
            """).fetchone()
        assert row is not None
        return {
            "total_calls": row[0],
            "total_tokens_in": row[1],
            "total_tokens_out": row[2],
            "total_tokens": row[1] + row[2],
            "avg_latency_ms": round(row[3], 1) if row[3] is not None else None,
            "error_rate_pct": round(row[4], 1) if row[4] is not None else None,
        }
