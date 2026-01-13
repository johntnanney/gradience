"""
Bench entry point (v0.1 scaffold).

In v0.1 commit 1 we only create the skeleton + config + reporting utilities.
The actual protocol wiring (train -> audit -> compress -> eval) lands later.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m gradience.bench.run_bench")
    p.add_argument("--config", required=True, help="Path to a bench config (yaml)")
    p.add_argument("--output", required=True, help="Output directory for bench artifacts")
    p.add_argument("--smoke", action="store_true", help="Run a tiny, fast bench (future)")
    p.add_argument("--ci", action="store_true", help="Fail if recommendations do not validate (future)")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Gradience Bench v0.1 scaffold")
    print(f"Config:  {args.config}")
    print(f"Output:  {out_dir}")
    print()
    print("Protocol not implemented yet (this is commit 1 scaffold).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())