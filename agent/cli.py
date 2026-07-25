"""
agent/cli.py
============
Command-line interface for the MitoAgent.

Examples
--------
Run the full pipeline on an Oroboros export:
    python -m agent.cli analyze data_samples/dataset_I.xlsx

Run only the identifiability analysis on a previously-fit dataset:
    python -m agent.cli identify data_samples/dataset_I.xlsx \\
        --params results/calib_dataset_I.json

Ask in natural language (offline keyword router by default):
    python -m agent.cli ask 'load "data_samples/dataset_I.xlsx" and calibrate'
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.orchestrator import MitoAgent
from agent.llm_driver   import NaturalLanguageDriver


def cmd_analyze(args):
    agent = MitoAgent(verbose=True)
    # Backward compatible: --full means publication_real_data; --fast means fast.
    mode = getattr(args, "mode", None) or ("publication_real_data" if args.full else "fast")
    fast = mode in {"smoke", "fast"}
    if args.fast and args.full:
        print("[mito-agent] both --fast and --full given; using --full/publication_real_data.")
    report = agent.run_pipeline(
        args.path,
        fast=fast,
        coverage_band=(float(args.coverage_band_low),
                        float(args.coverage_band_high)),
        fim_sloppy_threshold=float(args.fim_sloppy_threshold),
        chamber_index=args.chamber,
    )
    ok = bool(report.get("ok"))
    # Always persist the report if a path was given (a failed run still
    # produces a diagnostic report worth inspecting), but make the on-screen
    # message and exit code unambiguous about success vs failure.
    if args.out:
        agent.save_report(args.out)
        if ok:
            print(f"\nReport saved: {args.out}")
        else:
            err = report.get("error") or "pipeline did not complete"
            print(f"\n[mito-agent] PIPELINE FAILED: {err}", file=sys.stderr)
            print(f"[mito-agent] Diagnostic report (incomplete) written to: "
                  f"{args.out}", file=sys.stderr)
    elif not ok:
        err = report.get("error") or "pipeline did not complete"
        print(f"\n[mito-agent] PIPELINE FAILED: {err}", file=sys.stderr)
    return 0 if ok else 1


def cmd_identify(args):
    agent = MitoAgent(verbose=True)
    r = agent.load(args.path, chamber_index=args.chamber)
    if not r["ok"]:
        print(f"[mito-agent] LOAD FAILED: {r.get('message')}", file=sys.stderr)
        return 1
    pr = agent.preprocess()
    if not pr["ok"]:
        print(f"[mito-agent] PREPROCESS FAILED: {pr.get('message')}",
              file=sys.stderr)
        return 1
    if args.params and os.path.exists(args.params):
        params = json.load(open(args.params))["params"]
        agent.state.params = params
        print(f"Loaded params from {args.params}")
    else:
        print("Calibrating first (no --params supplied)...")
        cr = agent.calibrate(method="staged",
                             maxiter_stageA=20, maxiter_stageB=40)
        if not cr["ok"]:
            print(f"[mito-agent] CALIBRATION FAILED: {cr.get('message')}",
                  file=sys.stderr)
            return 1
    ir = agent.identifiability(method=args.method)
    if not ir.get("ok", True):
        print(f"[mito-agent] IDENTIFIABILITY FAILED: {ir.get('message')}",
              file=sys.stderr)
        return 1
    return 0


def cmd_ask(args):
    agent = MitoAgent(verbose=False)
    nl = NaturalLanguageDriver(agent, provider=args.provider)
    out = nl.ask(args.query)
    print(out)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="mito-agent",
                                  description="Mitochondrial bioenergetics AI agent.")
    sub = p.add_subparsers(dest="command")

    pa = sub.add_parser("analyze", help="Run full pipeline on a data file")
    pa.add_argument("path", help="Excel/CSV/NPY data file")
    pa.add_argument("--chamber", type=int, default=0,
                     help="Which chamber index to analyse")
    pa.add_argument("--full",   action="store_true",
                     help="Use slow / publication-quality settings")
    pa.add_argument("--fast",   action="store_true",
                     help="Use fast / diagnostic settings (default)")
    pa.add_argument("--mode", choices=["fast", "publication", "publication_real_data"],
                     default=None,
                     help="Run mode. publication is an alias for publication_real_data. For full batch real-data runs, prefer run_real_data.py.")
    pa.add_argument("--coverage-band-low",  type=float, default=0.80,
                     help="Configurable coverage warning threshold (low)")
    pa.add_argument("--coverage-band-high", type=float, default=0.95,
                     help="Configurable coverage warning threshold (high)")
    pa.add_argument("--fim-sloppy-threshold", type=float, default=1e15,
                     help="FIM raw-condition warning threshold")
    pa.add_argument("--out",    default=None,
                     help="Save JSON report to this path")
    pa.set_defaults(func=cmd_analyze)

    pi = sub.add_parser("identify", help="Run identifiability analysis only")
    pi.add_argument("path")
    pi.add_argument("--chamber", type=int, default=0)
    pi.add_argument("--method", choices=["fim", "profile", "both"],
                      default="fim")
    pi.add_argument("--params", default=None,
                      help="Pre-fitted params JSON")
    pi.set_defaults(func=cmd_identify)

    pq = sub.add_parser("ask", help="Natural-language query")
    pq.add_argument("query")
    pq.add_argument("--provider", choices=["auto", "anthropic", "offline"],
                      default="auto")
    pq.set_defaults(func=cmd_ask)

    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
