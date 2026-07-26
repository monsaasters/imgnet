"""
Allow `python -m imgnet` to work.
"""

from imgnet.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
