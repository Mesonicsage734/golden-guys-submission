"""Gate for changes to the submission: rules compliance, then a real strength check.

Canonical rules (re-fetch these before trusting a number in here, they change):
  https://aichessathon.com/docs/agent-contract.md
  https://aichessathon.com/docs/rules.md

Three checks, in order:
  1. lint + types clean (ruff, mypy)
  2. the packaged submission is under the 50 MB unzipped cap -- a hard fail here, not the
     warning harness/package.py prints on its own
  3. the working copy of agent.py (+ weights/) beats whatever was last committed, over a
     self-play arena match, by at least --min-score

Check 3 is a coarse filter, not a significance test: at the default game count its noise band
is wide, so treat a close score as "inconclusive, rerun with more games" rather than as proof
either way, and run many more games (make commit-gate GAMES=200) before trusting a result you're
about to upload.

Exit 0 only if every check that applies passes. Wired in as the pre-commit hook (see
scripts/pre-commit); run it by hand any time with `make commit-gate`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GAMES = 40
DEFAULT_MIN_SCORE = 0.55  # must clearly beat the last commit, not just edge past 50%


def run_text(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def run_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, cwd=ROOT, capture_output=True)


def check_lint_and_types() -> bool:
    ok = True
    for label, cmd in [
        ("ruff", ("uv", "run", "ruff", "check", ".")),
        ("mypy", ("uv", "run", "mypy")),
    ]:
        result = run_text(*cmd)
        if result.returncode != 0:
            print(f"[gate] {label} failed:\n{result.stdout}{result.stderr}")
            ok = False
    if ok:
        print("[gate] ruff + mypy clean")
    return ok


def check_size() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "submission.zip"
        result = run_text("uv", "run", "python", "-m", "harness.package", "--out", str(out))
        print(result.stdout, end="")
        if result.returncode != 0:
            print(f"[gate] package build failed:\n{result.stderr}")
            return False
        if "warning:" in result.stdout:
            print(
                "[gate] over the 50 MB unzipped cap -- that is a hard fail for this gate, "
                "not a warning; the platform will reject the upload outright"
            )
            return False
    print("[gate] under the 50 MB unzipped cap")
    return True


def _starter_agent_content() -> bytes | None:
    """The placeholder agent.py this repo shipped with, straight from git history.

    None if history doesn't have it (shallow clone, or agent.py was never committed).
    """
    added = run_text("git", "log", "--diff-filter=A", "--format=%H", "--follow", "--", "agent.py")
    if added.returncode != 0 or not added.stdout.strip():
        return None
    first_commit = added.stdout.strip().splitlines()[-1]
    show = run_bytes("git", "show", f"{first_commit}:agent.py")
    return show.stdout if show.returncode == 0 else None


def _head_is_unmodified_starter() -> bool:
    """True if HEAD's agent.py is still exactly the random-mover placeholder.

    There's no real agent committed yet in that case, even though the file is tracked --
    the same "nothing to compare against" situation the very-first-commit check exists for.
    """
    starter = _starter_agent_content()
    if starter is None:
        return False
    current = run_bytes("git", "show", "HEAD:agent.py")
    return current.returncode == 0 and current.stdout == starter


def extract_previous_submission(destination: Path) -> bool:
    """Copy whatever agent.py (+ weights/) HEAD has committed into `destination`.

    Returns False if there's nothing committed yet to compare against, so the very first
    commit of a real agent isn't blocked waiting for an opponent that doesn't exist. That
    covers both agent.py being untracked and it still being the unmodified starter --
    beating a uniformly random mover isn't a meaningful regression test either way.
    """
    listing = run_text("git", "ls-tree", "-r", "--name-only", "HEAD")
    if listing.returncode != 0:
        return False
    tracked = [line for line in listing.stdout.splitlines() if line]
    if "agent.py" not in tracked or _head_is_unmodified_starter():
        return False
    for name in tracked:
        if name != "agent.py" and not name.startswith("weights/"):
            continue
        show = run_bytes("git", "show", f"HEAD:{name}")
        if show.returncode != 0:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(show.stdout)
    return (destination / "agent.py").exists()


def check_strength(games: int, min_score: float) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        previous = Path(tmp)
        if not extract_previous_submission(previous):
            print("[gate] nothing committed yet to compare against -- skipping the "
                  "regression check for this commit")
            return True
        result = run_text(
            "uv", "run", "python", "-m", "harness.arena",
            "--agent", str(ROOT), "--opponent", str(previous), "--games", str(games),
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(
                f"[gate] the new version crashed, flagged, or made an illegal move against "
                f"the last committed version -- that alone fails the gate:\n{result.stderr}"
            )
            return False
        match = re.search(r"score (\d+(?:\.\d+)?)%", result.stdout)
        if not match:
            print("[gate] could not read a score out of the arena output")
            return False
        score = float(match.group(1)) / 100
        if score < min_score:
            print(
                f"[gate] scored {score:.1%} against the last committed version over {games} "
                f"games, need at least {min_score:.0%} -- not a confirmed improvement, "
                "do not commit"
            )
            return False
        print(f"[gate] scored {score:.1%} against the last committed version over {games} games")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()

    checks = [
        ("lint & types", check_lint_and_types()),
        ("size", check_size()),
        ("strength vs last commit", check_strength(args.games, args.min_score)),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"\n[gate] FAILED: {', '.join(failed)}. Do not commit.")
        sys.exit(1)
    print("\n[gate] all checks passed.")


if __name__ == "__main__":
    main()
