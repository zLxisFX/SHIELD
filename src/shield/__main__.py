from __future__ import annotations

import sys


def main() -> None:
    # Launch GUI if requested
    if "--gui" in sys.argv:
        # Remove flag so GUI doesn't see it (and so future parsing stays clean)
        sys.argv = [a for a in sys.argv if a != "--gui"]
        from shield.app.gui import main as gui_main
        raise SystemExit(gui_main())

    # Default: existing CLI behavior
    from shield.app import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
