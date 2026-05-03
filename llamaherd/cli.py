"""LlamaHerd CLI — start the proxy server."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="llamaherd",
        description="LlamaHerd — Herd your Ollama Cloud subscriptions. Multi-key proxy with load balancing, usage tracking, and a live dashboard.",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override bind host (default: from config or 127.0.0.1)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Override bind port (default: from config or 8399)",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="Override admin token (default: from config.yaml admin_token)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Import here so --help is fast
    from .proxy import load_config, main as proxy_main

    # Patch config path and overrides before starting
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


if __name__ == "__main__":
    main()