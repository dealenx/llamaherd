"""LlamaHerd CLI — agent-facing controls for managing the proxy."""

import argparse
import json
import sys

import httpx

from . import __tagline__, __version__

BANNER = r"""
    __    __                      __  __              __
   / /   / /___ _____ ___  ____ _/ / / /__  _________/ /
  / /   / / __ `/ __ `__ \/ __ `/ /_/ / _ \/ ___/ __  /
 / /___/ / /_/ / / / / / / /_/ / __  /  __/ /  / /_/ /
/_____/_/\__,_/_/ /_/ /_/\__,_/_/ /_/\___/_/   \__,_/
""".strip("\n")


def _base_url(args) -> str:
    host = getattr(args, "host", None) or "127.0.0.1"
    port = getattr(args, "port", None) or 8399
    return f"http://{host}:{port}"


def _token(args) -> str:
    """Get admin token from --token flag or config file."""
    # Check --token flag first (global arg)
    tok = getattr(args, "token", None)
    if tok:
        return tok
    # Check --admin-token flag (serve subcommand)
    tok = getattr(args, "admin_token", None)
    if tok:
        return tok
    # Try reading from config
    try:
        import yaml
        config_path = getattr(args, "config", "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("admin_token", "")
    except Exception:
        return ""


def _headers(args) -> dict:
    token = _token(args)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _api(args, method: str, path: str, json_body=None, params=None) -> dict:
    """Make an API call and return parsed JSON. Exits on error."""
    base = _base_url(args)
    url = f"{base}{path}"
    try:
        with httpx.Client(timeout=10) as client:
            if method == "GET":
                r = client.get(url, headers=_headers(args), params=params)
            elif method == "POST":
                r = client.post(url, headers=_headers(args), json=json_body)
            elif method == "PUT":
                r = client.put(url, headers=_headers(args), json=json_body)
            elif method == "PATCH":
                r = client.patch(url, headers=_headers(args), json=json_body)
            elif method == "DELETE":
                r = client.delete(url, headers=_headers(args))
            else:
                raise ValueError(f"Unknown method: {method}")
        if r.status_code >= 400:
            print(json.dumps({"error": f"HTTP {r.status_code}", "detail": r.text}, indent=2), file=sys.stderr)
            sys.exit(1)
        return r.json() if r.text else {}
    except httpx.ConnectError:
        print(json.dumps({"error": "connection_refused", "detail": f"Cannot connect to {base}"}), file=sys.stderr)
        sys.exit(1)


def _format_output(data, fmt="json", table_fn=None):
    """Output data in requested format."""
    if fmt == "json" or table_fn is None:
        print(json.dumps(data, indent=2))
    else:
        table_fn(data)


# ---- Clients commands ----

def cmd_clients_list(args):
    data = _api(args, "GET", "/admin/clients")
    if not isinstance(data, list):
        data = [data] if data else []
    _format_output(data, args.format, _clients_table)


def _clients_table(clients):
    if not clients:
        print("No clients.")
        return
    print(f"{'ID':<20} {'Label':<25} {'Token':<35} {'Token Limit':>12} {'Req Limit':>10} {'RPM':>5}")
    print("-" * 110)
    for c in clients:
        tok = c.get("token", "")
        tok_display = tok[:8] + "..." + tok[-4:] if len(tok) > 12 else tok
        dtl = c.get("daily_token_limit")
        drl = c.get("daily_request_limit")
        rpm = c.get("rpm_limit")
        print(f"{c['id']:<20} {c.get('label', ''):<25} {tok_display:<35} {str(dtl or '∞'):>12} {str(drl or '∞'):>10} {str(rpm or '∞'):>5}")


def cmd_clients_create(args):
    body = {
        "id": args.client_id,
        "label": args.label or args.client_id,
    }
    if args.notes:
        body["notes"] = args.notes
    if args.daily_token_limit is not None:
        body["daily_token_limit"] = args.daily_token_limit
    if args.daily_request_limit is not None:
        body["daily_request_limit"] = args.daily_request_limit
    if args.rpm_limit is not None:
        body["rpm_limit"] = args.rpm_limit
    result = _api(args, "POST", "/admin/clients", json_body=body)
    _format_output(result, args.format)


def cmd_clients_delete(args):
    result = _api(args, "DELETE", f"/admin/clients/{args.client_id}")
    _format_output(result, args.format)


def cmd_clients_regen(args):
    result = _api(args, "POST", f"/admin/clients/{args.client_id}/regenerate-token")
    _format_output(result, args.format)


def cmd_clients_update(args):
    body = {}
    if args.label is not None:
        body["label"] = args.label
    if args.notes is not None:
        body["notes"] = args.notes
    # For limits: --clear-limits sends null, otherwise use provided value
    if args.clear_limits:
        body["daily_token_limit"] = None
        body["daily_request_limit"] = None
        body["rpm_limit"] = None
    else:
        if args.daily_token_limit is not None:
            body["daily_token_limit"] = args.daily_token_limit
        if args.daily_request_limit is not None:
            body["daily_request_limit"] = args.daily_request_limit
        if args.rpm_limit is not None:
            body["rpm_limit"] = args.rpm_limit
    if not body:
        print(json.dumps({"error": "no changes specified"}), file=sys.stderr)
        sys.exit(1)
    result = _api(args, "PATCH", f"/admin/clients/{args.client_id}", json_body=body)
    _format_output(result, args.format)


# ---- Keys commands ----

def cmd_keys_list(args):
    data = _api(args, "GET", "/admin/keys")
    _format_output(data, args.format, _keys_table)


def _keys_table(keys):
    if not isinstance(keys, list):
        keys = [keys] if keys else []
    if not keys:
        print("No upstream keys.")
        return
    print(f"{'#':<3} {'Label':<25} {'Token':<15} {'Slots':>6} {'429s':>5} {'Plan':<15}")
    print("-" * 72)
    for i, k in enumerate(keys):
        tok = k.get("token_prefix", "")
        slots = f"{k.get('available_slots', '?')}/{k.get('max_concurrent', '?')}"
        print(f"{i:<3} {k.get('label', ''):<25} {tok:<15} {slots:>6} {k.get('total_429s', 0):>5} {k.get('plan', ''):<15}")


# ---- Status command ----

def cmd_status(args):
    data = _api(args, "GET", "/admin/status")
    _format_output(data, args.format, _status_table)


def _status_table(data):
    keys = data.get("keys", [])
    if keys:
        print("Upstream Keys:")
        _keys_table(keys)
    totals = data.get("totals", {})
    if totals:
        print(f"\nTotals: {totals.get('total_calls', 0)} calls, {totals.get('total_tokens', 0)} tokens")


# ---- Usage command ----

def cmd_usage(args):
    params = {}
    if args.days:
        params["days"] = args.days
    if args.start_date:
        params["start_date"] = args.start_date
    if args.end_date:
        params["end_date"] = args.end_date
    if args.client:
        params["client"] = args.client
    if args.model:
        params["model"] = args.model
    data = _api(args, "GET", "/admin/totals", params=params)
    _format_output(data, args.format)


# ---- Models command ----

def cmd_models(args):
    data = _api(args, "GET", "/admin/models")
    _format_output(data, args.format, _models_table)


def _models_table(data):
    models = data.get("models", [])
    if not models:
        print("No models discovered.")
        return
    print(f"{'Model':<35} {'Context':>10} {'Keys':>5}")
    print("-" * 55)
    for m in models:
        cl = m.get('context_length') or '?'
        ao = m.get('available_on') or '?'
        print(f"{m['id']:<35} {str(cl):>10} {str(ao):>5}")
    print(f"\nTotal: {data.get('count', len(models))} models")


# ---- Branding command ----

def cmd_banner(args):
    print(BANNER)
    print(f"\n{__tagline__}")


# ---- Build the parser ----

def build_parser():
    parser = argparse.ArgumentParser(
        prog="llamaherd",
        description=f"LlamaHerd — {__tagline__}",
    )
    parser.add_argument("--version", action="version", version=f"llamaherd {__version__}")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--host", default=None, help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=None, help="API port (default: 8399)")
    parser.add_argument("--token", "-t", default=None, help="Admin token (or read from config)")
    parser.add_argument("--format", "-f", choices=["json", "table"], default="json",
                        help="Output format (default: json, for agents)")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- clients ---
    clients = sub.add_parser("clients", help="Manage client keys")
    clients_sub = clients.add_subparsers(dest="subcommand", help="Client operations")

    # clients list
    cl = clients_sub.add_parser("list", help="List all clients")
    cl.set_defaults(func=cmd_clients_list)

    # clients create
    cc = clients_sub.add_parser("create", help="Create a client key")
    cc.add_argument("client_id", help="Unique client ID (alphanumeric, dashes ok)")
    cc.add_argument("--label", "-l", help="Human-readable label")
    cc.add_argument("--notes", help="Notes about this client")
    cc.add_argument("--daily-token-limit", type=int, default=None, help="Max tokens per day (null=unlimited)")
    cc.add_argument("--daily-request-limit", type=int, default=None, help="Max requests per day (null=unlimited)")
    cc.add_argument("--rpm-limit", type=int, default=None, help="Max requests per minute (null=unlimited)")
    cc.set_defaults(func=cmd_clients_create)

    # clients update
    cu = clients_sub.add_parser("update", help="Update a client key")
    cu.add_argument("client_id", help="Client ID to update")
    cu.add_argument("--label", "-l", default=None, help="New label")
    cu.add_argument("--notes", default=None, help="New notes")
    cu.add_argument("--daily-token-limit", type=int, default=None, help="Max tokens per day")
    cu.add_argument("--daily-request-limit", type=int, default=None, help="Max requests per day")
    cu.add_argument("--rpm-limit", type=int, default=None, help="Max requests per minute")
    cu.add_argument("--clear-limits", action="store_true", help="Set all limits to null (unlimited)")
    cu.set_defaults(func=cmd_clients_update)

    # clients delete
    cd = clients_sub.add_parser("delete", help="Delete a client key")
    cd.add_argument("client_id", help="Client ID to delete")
    cd.set_defaults(func=cmd_clients_delete)

    # clients regenerate-token
    cr = clients_sub.add_parser("regenerate-token", help="Regenerate a client's API token")
    cr.add_argument("client_id", help="Client ID")
    cr.set_defaults(func=cmd_clients_regen)

    # --- keys ---
    keys = sub.add_parser("keys", help="List upstream Ollama Cloud keys")
    keys.set_defaults(func=cmd_keys_list)

    # --- status ---
    status = sub.add_parser("status", help="Show proxy status and key health")
    status.set_defaults(func=cmd_status)

    # --- usage ---
    usage = sub.add_parser("usage", help="Show usage totals")
    usage.add_argument("--days", type=int, default=None, help="Last N days")
    usage.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD)")
    usage.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    usage.add_argument("--client", default=None, help="Filter by client ID")
    usage.add_argument("--model", default=None, help="Filter by model")
    usage.set_defaults(func=cmd_usage)

    # --- models ---
    models = sub.add_parser("models", help="List discovered models")
    models.set_defaults(func=cmd_models)

    # --- banner ---
    banner = sub.add_parser("banner", help="Print the LlamaHerd ASCII banner")
    banner.set_defaults(func=cmd_banner)

    # --- serve (run the proxy) ---
    serve = sub.add_parser("serve", help="Start the proxy server")
    serve.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    serve.add_argument("--admin-token", default=None, help="Override admin token from config")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        # Import and start the proxy server
        from .proxy import main as proxy_main, load_config
        import os
        if args.config:
            os.environ["LLAMAHERD_CONFIG"] = args.config
        if args.admin_token:
            os.environ["LLAMAHERD_ADMIN_TOKEN"] = args.admin_token
        if args.host:
            os.environ["LLAMAHERD_HOST"] = args.host
        if args.port:
            os.environ["LLAMAHERD_PORT"] = str(args.port)
        proxy_main()
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()