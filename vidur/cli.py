"""Scriptable CLI and interactive terminal entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime

from vidur.collectors import earthquakes, news, weather
from vidur.agent_reach import AgentReachError, doctor, search_platform, search_twitter
from vidur.storage import init_db, list_items, stats


def _render(items: list[dict], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(items, indent=2, default=str)); return
    if not items:
        print("No intelligence items found."); return
    for item in items:
        when = item.get("collected_at", "")[:19].replace("T", " ")
        print(f"[{item.get('category', '?'):<7}] {item.get('severity', 'info').upper():<6} {when}  {item.get('title', '')}")
        if item.get("summary"): print(f"  {item['summary'][:220]}")
        if item.get("source_url"): print(f"  {item['source_url']}")


def _report(items: list[dict], target: str | None) -> None:
    title = target or "Stored intelligence"
    print(f"\nVIDUR SITUATION BRIEF — {title}\n{'=' * 58}")
    if not items:
        print("No matching intelligence is stored. Run `vidur collect` first."); return
    counts = Counter(item["category"] for item in items)
    risks = Counter(item["severity"] for item in items)
    print("Coverage: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("Severity: " + ", ".join(f"{k} {v}" for k, v in risks.items()))
    print("\nKey items:")
    for item in items[:10]: print(f"- [{item['severity'].upper()}] {item['title']}")
    print("\nAssessment: This brief summarizes public-source data. Verify all claims with primary sources before action.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vidur", description="Terminal-native situational awareness workspace")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="create Vidur's local database")
    sub.add_parser("tui", help="open the interactive research terminal")
    sub.add_parser("status", help="show local workspace status")
    sub.add_parser("sources", help="show Agent Reach source-channel status")
    x_search = sub.add_parser("x-search", help="search X through the active Agent Reach backend")
    x_search.add_argument("query")
    x_search.add_argument("--limit", type=int, default=10)
    x_search.add_argument("--json", action="store_true")
    source_search = sub.add_parser("source-search", help="search a read-only Agent Reach channel")
    source_search.add_argument("platform", choices=("x", "reddit", "github", "youtube"))
    source_search.add_argument("query")
    source_search.add_argument("--limit", type=int, default=10)
    source_search.add_argument("--json", action="store_true")
    collect = sub.add_parser("collect", help="collect public-source intelligence")
    collectors = collect.add_subparsers(dest="collector", required=True)
    quake = collectors.add_parser("earthquakes", help="collect USGS earthquake events")
    quake.add_argument("--min-magnitude", type=float, default=4.5)
    meteo = collectors.add_parser("weather", help="collect current weather")
    meteo.add_argument("location")
    headline = collectors.add_parser("news", help="collect public RSS news results")
    headline.add_argument("query"); headline.add_argument("--limit", type=int, default=20)
    intel = sub.add_parser("intel", help="inspect locally stored intelligence")
    intel_sub = intel.add_subparsers(dest="intel_command", required=True)
    for name in ("list", "search"):
        p = intel_sub.add_parser(name); p.add_argument("query", nargs="?" if name == "search" else "*")
        p.add_argument("--category"); p.add_argument("--limit", type=int, default=30); p.add_argument("--json", action="store_true")
    report = sub.add_parser("report", help="generate a local situation brief")
    report.add_argument("target", nargs="?"); report.add_argument("--category"); report.add_argument("--limit", type=int, default=50)
    timeline = sub.add_parser("timeline", help="show intelligence chronologically")
    timeline.add_argument("--category"); timeline.add_argument("--limit", type=int, default=50); timeline.add_argument("--json", action="store_true")
    watch = sub.add_parser("watch", help="periodically refresh a safe public collector")
    watch.add_argument("collector", choices=("earthquakes",)); watch.add_argument("--interval", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.command:
        args.command = "tui"
    if args.command == "init": print(f"Initialized {init_db()}"); return
    init_db()
    if args.command == "tui":
        from vidur.tui import run
        run(); return
    if args.command == "status":
        data = stats(); print(f"Database: {data['database']}\nItems: {data['total']}\nLatest: {data['latest'] or '—'}")
        print(f"Source posts: {data['source_items']}")
        print("Categories: " + (", ".join(f"{r['category']}={r['count']}" for r in data['categories']) or "—")); return
    if args.command == "sources":
        try:
            for channel in doctor():
                backend = channel.backend or "backend not verified"
                print(f"{channel.platform:<16} {channel.status:<10} {backend}")
                if channel.message: print(f"  {channel.message}")
        except AgentReachError as error:
            print(f"Agent Reach unavailable: {error}", file=sys.stderr)
            raise SystemExit(1)
        return
    if args.command == "x-search":
        try:
            items = search_twitter(args.query, args.limit)
        except AgentReachError as error:
            print(f"X search failed: {error}", file=sys.stderr)
            raise SystemExit(1)
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2, default=str))
        else:
            for item in items:
                print(f"{item['author']} · {item['published_at']} · {item['backend']}")
                print(item["body"])
                print(item["source_url"])
                if item["media"]: print(f"{len(item['media'])} attachment(s) · view in `vidur tui`")
                print()
        return
    if args.command == "source-search":
        try:
            items = search_platform(args.platform, args.query, args.limit)
        except AgentReachError as error:
            print(f"{args.platform.upper()} search failed: {error}", file=sys.stderr)
            raise SystemExit(1)
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2, default=str))
        else:
            for item in items:
                print(f"{item['platform'].upper()} · {item['author']} · {item['published_at']} · {item['backend']}")
                print(item["body"])
                print(item["source_url"])
                print()
        return
    if args.command == "collect":
        try:
            if args.collector == "earthquakes": items = earthquakes(args.min_magnitude)
            elif args.collector == "weather": items = weather(args.location)
            else: items = news(args.query, args.limit)
            print(f"Collected {len(items)} item(s).")
            _render(items)
        except Exception as error:
            print(f"Collection failed: {error}", file=sys.stderr); raise SystemExit(1)
        return
    if args.command == "watch":
        try:
            while True:
                items = earthquakes(); print(f"{datetime.now().isoformat(timespec='seconds')}: collected {len(items)} earthquake event(s)")
                time.sleep(max(30, args.interval))
        except KeyboardInterrupt: print("\nWatch stopped.")
        return
    if args.command == "intel":
        query = " ".join(args.query) if isinstance(args.query, list) else args.query
        _render(list_items(args.category, query, args.limit), args.json); return
    if args.command == "timeline":
        _render(list_items(args.category, limit=args.limit), args.json); return
    if args.command == "report":
        _report(list_items(args.category, args.target, args.limit), args.target)


if __name__ == "__main__":
    main()
