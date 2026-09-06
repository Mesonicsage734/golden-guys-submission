# Working in this repo

This is a starter for AI Chessathon, a chess-engine competition. The deliverable is one file,
`agent.py`, exposing `get_move(fen, time_left_ms) -> str`. It gets zipped and uploaded, and the
platform plays it against other people's agents on a fixed cadence.

## Commit gate -- read this before changing agent.py

Compliance with the published rules is the floor, not a nice-to-have, and it is checked
*before* any change is judged on strength. Concretely, for any change that touches `agent.py`
or `weights/`:

1. Confirm the baseline still holds. Re-fetch the two canonical URLs below if you have not
   checked them recently in this session, and re-read the quick reference here against them.
   A change built on a stale understanding of the rules is not safe to commit no matter how it
   scores.
2. Run `make commit-gate` (this is also wired in as the pre-commit hook, see below, so it runs
   automatically). It checks, in order:
   - lint and types clean (`ruff`, `mypy`)
   - the packaged submission is under the 50 MB unzipped cap -- treated as a hard fail here,
     not the warning `harness/package.py` prints on its own, because the platform will reject
     an oversize upload outright
   - the working copy of `agent.py` (+ `weights/` if present) beats whatever is currently
     committed, in a self-play arena match, by a real margin (default: score >= 55% over 40
     games against the exact last-committed version, extracted straight from `git show HEAD:...`)
3. Do not commit if any of the three fail. Do not weaken the gate (lower `--min-score`, cut
   `--games`, edit `scripts/commit_gate.py` to be easier) to force a change through -- fix the
   change or don't commit it.
4. The default 40-game check is a coarse filter, not a significance test: at that sample size a
   55% score is well within noise of a true 50%. Treat anything close as inconclusive and rerun
   with more games (`make commit-gate GAMES=200`) rather than trusting the first number, and run
   a much larger comparison before anything you are about to actually upload to the platform,
   not just before a local commit.
5. On the very first commit of a real agent there is nothing previous to compare against, so the
   gate skips step 3's regression check automatically -- lint, types and size still apply. This
   also covers the case where `agent.py` is tracked but HEAD's copy is still byte-identical to
   the unmodified starter random-mover: beating a uniformly random mover isn't a meaningful
   regression test, so a neutral, non-strength-changing commit (e.g. adding a safety-net wrapper
   around the starter before any real search exists) can land without a fabricated "improvement"
   over random play. The moment HEAD's `agent.py` differs from the starter, the regression check
   is back in force for every commit after that.

This is enforced, not just documented: `make setup` installs `scripts/pre-commit` as the git
pre-commit hook, and it runs the full gate whenever `agent.py` or `weights/` are staged (a
docs-only commit just gets lint+types, not a 40-game match). Never bypass it with
`git commit --no-verify` -- if the gate is wrong for a situation, fix `scripts/commit_gate.py`
deliberately and say so, don't route around it silently.

## Read the rules from the source

The competition rules and the agent contract live on the site and change. Fetch them before you
answer anything about limits, deadlines, or what is allowed:

- https://aichessathon.com/docs/agent-contract.md
- https://aichessathon.com/docs/rules.md

The quick reference below is a convenience for common questions. The two URLs are canonical
and they change, so fetch them before you rely on a number.

## The contract, in one place

- `agent.py` at the root of the zip, not inside a folder. The platform does `import agent`.
- `get_move(fen: str, time_left_ms: int) -> str` returning UCI, `e2e4` or `e7e8q`.
- Your colour is the side to move in the fen. There is no other input.
- The process starts once per game and stays alive between your moves. Module state survives to
  your next move in the same game, never to the next game.
- Import time has a 90 second budget before the clock starts. Load weights there.
- 120 s + 0.5 s per move, per side, on wall time. One core of an AMD EPYC 9V74 at 2.60 GHz, 2 GB,
  no network, no GPU.
- Illegal move, malformed output, crash, out of memory, or flag fall loses that game. A move
  reply over 4 KB counts as illegal. 300 plies without a result goes to material adjudication.
- Everything in the zip together stays under 50 MB unzipped.
- Ten uploads per team per day, and the latest one that passed validation is the one that plays.
- Rated games start from curated opening positions, not the standard start. The set is not
  published.
- Your process is suspended while the opponent thinks, so nothing you leave running between your
  own moves gets any CPU. Do your searching inside `get_move`. Two of your games can run at once,
  in separate containers.

## Things that break agents here

- The filesystem is read-only apart from 256 MB at `/tmp`. `HOME` and every cache path already
  point there; do not write anywhere else.
- No network at all. Nothing downloads at runtime. Weights ship inside the zip.
- One core. `torch.set_num_threads(1)`. More threads lose time rather than winning it.
- Your zip is first on `sys.path`. Never name a file after a module you import: `chess.py`,
  `types.py`, `random.py` will shadow the real one and the failure will look unrelated.
- The environment is fixed. The platform preinstalls torch 2.13 (CPU), numpy 2.5, python-chess
  1.11, onnxruntime 1.29 and numba 0.67 and installs nothing else. A `requirements.txt` is
  ignored, so an import outside that stack crashes on the platform even when it works locally.
  Additions can be requested at hello@aichessathon.com.
- Native binaries in the zip are rejected. Ship Python source. Model weights like `.onnx`,
  `.safetensors` and `.pt` are fine. The network has to be one the team trained, and training it
  on positions an engine labelled is a normal way to do that.
- numba is how Python gets fast here. Warm every jitted function once at import so compilation
  lands in the init budget, not on the clock. Cython does not work on the platform.
- `print` is safe. The runner points file descriptor 1 at stderr before importing the agent, so
  nothing you write can corrupt the protocol. It is discarded in rated games and shown in the
  validation log.

## Do not

- Do not ship someone else's engine or someone else's network. Stockfish, Lc0, Maia, a port or
  translation of one, and a published net that was fine-tuned or re-exported all count. This is
  checked after games are played, not only at upload.
- Do not ship a table of engine moves or evaluations for the agent to look up while it plays.
  That is an engine in another shape. Opening books and endgame tablebases are fine.
- Do not add network calls, subprocess calls to external binaries, or anything that reads outside
  the agent directory and `/tmp`.
- Do not obfuscate. What ships has to be source a judge can read.
- Do not edit `harness/`. It mirrors the platform's protocol and clock. Changing it makes local
  results meaningless.

## Verify

```
make play      # one game against a baseline, real time control
make arena     # 20 fast games against a baseline, with a score
make zip       # build submission.zip with agent.py at the root
make gate      # ruff, mypy, and two games that have to finish cleanly
make commit-gate  # the full pre-commit policy: lint, size, and a strength check vs HEAD
```

Nothing here decides whether an upload is accepted. The platform validates on upload and writes a
log to the dashboard; that log is the authority. The harness exists so local games are honest.

## Style

Python 3.12, type-annotated, ruff and mypy strict clean. Keep `agent.py` readable: it is the
thing a judge reads if your games get flagged, and the thing you have to explain at the final.
