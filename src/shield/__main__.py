"""
Allows running SHIELD as:
  python -m shield [args...]

This simply forwards to shield.app.main().
"""

from __future__ import annotations

from .app import main

if __name__ == "__main__":
    main()
