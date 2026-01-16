from __future__ import annotations

from pathlib import Path
import sys

# Print hour lines that contain any of these tokens (edit if needed)
TOKENS = ("T09:", "T10:", "T11:", "T12:", "T13:", "T14:")

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools\\print_plan_window.py <plan_txt_path>")
        return 2

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        return 2

    print(p)
    lines = p.read_text(encoding="utf-8").splitlines()

    print("\n--- HEADER (first 25 lines) ---")
    for L in lines[:25]:
        print(L)

    print("\n--- WINDOW (matching lines) ---")
    hits = 0
    for L in lines:
        if any(tok in L for tok in TOKENS):
            print(L)
            hits += 1

    if hits == 0:
        print("(No matching lines. If your plan lines don’t use 'T10:' format, tell me what the timestamps look like.)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())