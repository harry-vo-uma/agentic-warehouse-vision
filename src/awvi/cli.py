"""Command line entry point: `awvi <command>`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_settings
from .index.store import EventIndex
from .pipeline import Pipeline


def _cmd_run(args) -> int:
    pipe = Pipeline()
    scenarios = Path(args.scenarios)
    if scenarios.is_dir():
        events = pipe.process_all(scenarios)
    else:
        events = pipe.process_scenario(json.loads(scenarios.read_text()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([json.loads(e.model_dump_json()) for e in events], indent=2))

    report = pipe.report.as_dict()
    print(json.dumps(report, indent=2))
    print(f"\n{len(events)} events written to {out}")
    for e in events:
        if e.disposition.value == "suppressed":
            continue
        print(f"  [{e.disposition.value:>9}] {e.severity.value:<8} {e.kind.value:<20} conf={e.confidence:.2f}  {e.summary}")
    return 0


def _cmd_query(args) -> int:
    events_path = Path(args.events)
    if not events_path.exists():
        print(f"no events file at {events_path}; run `awvi run` first", file=sys.stderr)
        return 1
    from .schemas import Event

    index = EventIndex()
    index.add_many([Event.model_validate(e) for e in json.loads(events_path.read_text())])
    resp = index.answer(args.question, top_k=args.top_k)
    print(resp.answer)
    print(f"\n({len(resp.hits)} hits in {resp.latency_ms:.0f}ms)")
    for h in resp.hits:
        e = h.event
        print(f"  {h.score:.3f}  [{e.event_id}] {e.kind.value:<20} {e.t_start:7.1f}s  {e.summary}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("awvi.api.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_info(args) -> int:
    from .agents.graph import langgraph_available
    from .nim.client import get_client
    from .perception.deepstream import DeepStreamSource

    print(json.dumps({
        "nim_mode": get_client().stats()["mode"],
        "nim_base_url": get_settings().nim.base_url,
        "vlm_model": get_settings().nim.vlm_model,
        "deepstream_runtime": DeepStreamSource.available(),
        "langgraph": langgraph_available(),
        "escalation": get_settings().escalation.__dict__,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("awvi", description="Agentic Warehouse Vision Intelligence")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process scenarios end to end")
    r.add_argument("--scenarios", default="data/scenarios")
    r.add_argument("--out", default="data/events.json")
    r.set_defaults(func=_cmd_run)

    q = sub.add_parser("query", help="ask a natural-language question over events")
    q.add_argument("question")
    q.add_argument("--events", default="data/events.json")
    q.add_argument("--top-k", type=int, default=5)
    q.set_defaults(func=_cmd_query)

    s = sub.add_parser("serve", help="run the operator API + UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)

    i = sub.add_parser("info", help="show resolved runtime configuration")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
