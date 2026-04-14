import subprocess
from typing import Tuple, Optional


def cmd(*args, cwd: Optional[str] = None, debug: bool = False) -> Tuple[str, int]:
    """Runs a command in the terminal and returns captured stdout text and exit code."""
    if debug:
        print(f"EXECUTING {' '.join(map(str, args))}")

    out = subprocess.run(args, check=False, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return out.stdout.decode("utf-8").rstrip(), out.returncode
