"""CLI entrypoint for `python3 -m autocorrupt`."""
import sys
from autocorrupt.daemon.slice_31_cli_and_verifier import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
