from __future__ import annotations

import argparse
import asyncio
import json
import sys

from llmserve.bench import (
    BenchSettings,
    LocalServer,
    format_table,
    format_verification,
    run_ablation,
    run_workload,
    summarize,
    verify_cache,
    write_results,
)
from llmserve.config import Config, ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmserve")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--local-files-only", action="store_true")

    bench = sub.add_parser("bench")
    bench.add_argument("--url", default=None)
    bench.add_argument("--requests", type=int, default=60)
    bench.add_argument("--rate", type=float, default=6.0)
    bench.add_argument("--max-tokens", type=int, default=64)
    bench.add_argument("--duplicate-fraction", type=float, default=0.2)
    bench.add_argument("--near-duplicate-fraction", type=float, default=0.1)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--model", default="auto")
    bench.add_argument("--request-timeout", type=float, default=None)
    bench.add_argument("--ablation", action="store_true")
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--verify-cache", action="store_true")
    bench.add_argument("--thresholds", default="")
    bench.add_argument("--out", default=None)
    bench.add_argument("--local-files-only", action="store_true")
    return parser


def command_serve(config: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from llmserve.api import build_gateway, create_app

    host = args.host or config.server.host
    port = args.port or config.server.port
    app = create_app(config, gateway=build_gateway(config, local_files_only=args.local_files_only))
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    return 0


def command_bench(config: Config, args: argparse.Namespace) -> int:
    if args.request_timeout:
        config = config.with_overrides({"server": {"request_timeout_s": args.request_timeout}})
    settings = BenchSettings(
        requests=args.requests,
        rate=args.rate,
        max_tokens=args.max_tokens,
        duplicate_fraction=args.duplicate_fraction,
        near_duplicate_fraction=args.near_duplicate_fraction,
        seed=args.seed,
        model=args.model,
        timeout_s=config.server.request_timeout_s + 60.0,
    )
    if args.verify_cache:
        thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
        report = verify_cache(config, settings, thresholds, local_files_only=args.local_files_only)
        print(format_verification(report))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            print(f"\nwrote {args.out}")
        return 0
    if args.ablation:
        results = run_ablation(config, settings, repeats=args.repeats, local_files_only=args.local_files_only)
    elif args.url:
        results = [asyncio.run(run_workload(args.url, settings, "all features"))]
    else:
        with LocalServer(config, local_files_only=args.local_files_only) as server:
            results = [asyncio.run(run_workload(server.url, settings, "all features"))]

    print(format_table(results))
    ratios = summarize(results)
    if ratios:
        print()
        for name, value in ratios.items():
            print(f"{name}: {value}")
    if args.out:
        write_results(args.out, results, settings)
        print(f"\nwrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = Config.from_toml(args.config)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    if args.command == "serve":
        return command_serve(config, args)
    return command_bench(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
