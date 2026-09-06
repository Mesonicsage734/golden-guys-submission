# Research: building a strong engine for AI Chessathon

Sources: the four research videos and the procrastination video the user linked, chessprogramming.org,
and this repo's own `docs/IDEAS.md` plus the canonical rules at aichessathon.com (fetched
2026-09-06 — re-check before relying on a number, per `AGENTS.md`). Every recommendation below is
filtered through the actual competition constraints, not general chess-engine advice.

## 1. The constraints that shape every decision

Pulled from `harness/rules.py`, `AGENTS.md`, and the live `aichessathon.com/docs/rules.md` /
`agent-contract.md`:

| Constraint | Value | Consequence |
|---|---|---|
| Clock | 120s + 0.5s/move, per side | Time management is a first-class feature, not an afterthought |
| Init budget | 90s, before the clock starts | Load weights / JIT-warm here, not in `get_move` |
| Hardware | 1 core, AMD EPYC 9V74 @ 2.6 GHz, 2 GB RAM, no GPU, no network | Pure-Python node counts are in the thousands, not millions; memory-hungry nets are out |
| Submission | `agent.py` at zip root, ≤50 MB unzipped, ≤10 uploads/day | Anything you ship (weights, book, tablebase) competes for the same 50 MB |
| Move reply | ≤4 KB, valid UCI, or you lose the game | Never let a malformed/oversized string escape `get_move` |
| Ply cap | 300 plies → material adjudication | A won-but-shuffling position can be thrown away by 50-move/repetition rules if you're not tracking them |
| Preinstalled stack | Python 3.12, `chess`, `numpy`, `numba`, `torch` (CPU), `onnxruntime` — fixed versions, nothing else installable | Design around exactly these; a `requirements.txt` is ignored on the platform |
| Filesystem | Read-only except `/tmp` (256 MB, wiped per game) | Never write elsewhere; caches (`HOME`, `TORCH_HOME`) already point at `/tmp` |
| Banned | Third-party engines/nets (Stockfish, Lc0, Maia, forks, fine-tunes), runtime move/eval lookup tables (an "engine in another shape") | Opening books and endgame tablebases are explicitly **allowed** — they're not the same thing as a move-lookup engine |
| Judged | Source must be readable, no obfuscation | Optimize for correctness and clarity, not cleverness that can't be explained at the final |

The practical upshot: this is a **classical search + light evaluation** competition, not a
"who has the biggest net" competition. 2 GB RAM and one CPU core rule out anything but a small
NNUE-sized network, and pure-Python search depth is the scarce resource everything else should be
spent protecting.

## 2. Search: where the points actually come from

Consistent across `docs/IDEAS.md`, chessprogramming.org, and both Sebastian Lague videos — the
search stack, roughly in ROI order for a single Python core:

1. **Negamax with alpha-beta pruning.** The baseline algorithm. Everything else is about making
   alpha-beta cut more branches or run more plies in the same wall time.
2. **Move ordering.** Alpha-beta's pruning power depends entirely on searching good moves first.
   - Try the transposition-table move first, if one exists.
   - Order captures by MVV-LVA (most valuable victim, least valuable attacker).
   - Killer-move and history heuristics for quiet moves are the next upgrade once the basics work.
3. **Iterative deepening.** Search depth 1, then 2, then 3… keeping the best move from the last
   *completed* depth. Two benefits that matter more here than in a compiled engine: it gives you
   ordering data for free (the previous depth's best move seeds the next), and it gives you a
   legal move to return the instant your time budget runs out — critical since a flag loss forfeits
   the game outright.
4. **Quiescence search.** Extend search at leaf nodes through captures (and ideally checks) before
   evaluating. Without it, your evaluation gets measured on positions mid-exchange and misjudges
   material that's about to be recaptured — the single biggest source of "engine blunders a piece"
   bugs in a from-scratch engine.
5. **Transposition table.** A dict keyed on a Zobrist hash (`chess.polyglot.zobrist_hash(board)`,
   or `board._transposition_key()` per `docs/IDEAS.md`) storing depth, score, bound type, and best
   move. Worth roughly a ply of effective depth for free. Because the process survives between your
   own moves (see §5), this table can persist across the whole game, not just one search call —
   just cap its size so it can't blow the 2 GB budget over a long game.
6. **Later-stage extras** (worth building only after 1–5 are solid and measured): null-move
   pruning, late move reductions (LMR — the technique Sebastian Lague's video spends real time on),
   aspiration windows, search extensions for checks/single-reply positions.

Sebastian Lague's *Coding Adventure: Making a Better Chess Bot* is the most directly applicable
video here: it walks the exact upgrade path above (bitboards → magic-bitboard move generation →
transposition tables/Zobrist hashing → killer moves → LMR → deployment/testing against real
opposition), in a from-scratch engine under similar "no existing engine code" rules to this
competition's community challenge format. Bartek Spitza's *The Fascinating Programming of a Chess
Engine* covers the same first two rungs (bitboard-style board representation, minimax with
alpha-beta) at a gentler pace — good as the on-ramp before Lague's video.

**Numba is the multiplier, not the strategy.** The platform preinstalls it specifically because
pure Python is slow enough that JIT-compiling the move generator and evaluation function meaningfully
raises the achievable search depth. `baselines/numba` in this repo shows the required pattern: warm
every jitted function once at import time (inside the 90s init budget, with the exact argument types
`get_move` will actually use — numba compiles per call signature) so compilation never eats into the
clock. Note it barely beats `baselines/minimax` on its own: jitting a shallow search wins nothing.
The gain shows up only once the freed-up time is spent on *depth* or a better evaluation.

## 3. Evaluation: how much model do you actually need

- **Material + piece-square tables (PST) is a real, competitive baseline** — it beats both
  `baselines/greedy` and `baselines/minimax` per `docs/IDEAS.md`'s own ladder, and it should be the
  *first* evaluation you write, both because it's cheap to call millions of times and because it
  gives you a reference point to prove a fancier evaluation is actually better than.
- **A learned evaluation is optional, and sized very differently than a "real" chess net.** With
  2 GB RAM, one CPU core, and no GPU:
  - An NNUE-style network quantized to int8/int16 (roughly 1–40 MB) is fast enough to call inside a
    search loop and fits comfortably in the 50 MB submission cap.
  - A full fp32 convolutional net (AlphaZero/Leela-style) manages only a few hundred evaluations per
    second on one core — that's a *policy* model's budget, not something you can call at every leaf
    of an alpha-beta search. This category is a trap for this competition specifically, however
    well it performs on GPU hardware elsewhere.
  - Export to ONNX and run with `onnxruntime` rather than raw `torch`: startup is faster and
    single-core inference is competitive, per `docs/IDEAS.md`.
  - Batch leaf evaluations from one search pass into a single inference call instead of one at a
    time — the difference matters at this evaluation volume.
- **Training data is entirely a pre-upload concern.** There's no network at runtime, so all
  training happens on your own machine beforehand: public game databases, self-play against your
  own earlier versions, and labelling positions with an existing engine are all explicitly allowed —
  the rule only bans *shipping* another engine or its weights, not learning from one offline. The
  net you ship has to be one your team actually trained.
- **The WIRED/GothamChess video is the "why" more than the "how."** It's a good gut-check on the
  ceiling classical search hits against something like Stockfish (Levy Rozman is crushed in 34
  moves), but it's aimed at a general audience and doesn't teach implementation detail — treat it as
  motivation for *why* search depth and evaluation quality compound, not as a technical reference.

## 4. Time management (the "loses for free" list starts here)

- A flag loss forfeits the game regardless of position — per `docs/IDEAS.md`, "it is the most
  common self-inflicted" loss, and it's a total loss regardless of how good the engine is otherwise.
- Budget per move from the actual clock you were handed, not a constant:
  `time_left_ms / max(20, expected_moves_left)` is a reasonable starting formula.
- Check the clock **inside** the search, not just between moves — iterative deepening makes this
  natural, since you can bail after any completed depth and still return that depth's best move.
- Leave a real safety margin: the referee measures wall time and the watchdog (`WATCHDOG_GRACE_MS`
  in `harness/rules.py`) does not forgive a late reply.
- The process stays alive between your own moves (state on a module or closure survives), but it is
  fully suspended while the opponent thinks — so all computation has to happen inside `get_move`
  itself, not on a background thread you hoped would keep running.

## 5. State worth keeping across moves

Because the agent process persists for the whole game (not just one call), two things are worth
carrying forward, per `docs/IDEAS.md`:

- **The transposition table.** Don't rebuild it from scratch every move.
- **Position history for draw detection.** The referee claims threefold repetition automatically —
  if your engine is winning and shuffles into a repeated position without knowing it, it throws away
  a win it never needed to concede.

An opening book is worth less here than the general chess-programming literature suggests: rated
games start from curated, unpublished positions rather than the standard starting position, so a
book keyed on move one is frequently already irrelevant. Test with `make play FEN=...` on
unfamiliar positions and spend the saved effort on search instead.

## 6. Engineering pitfalls specific to this platform

Straight from `AGENTS.md` / `docs/IDEAS.md` — these are platform-specific failure modes, not
general chess bugs, and they're exactly the kind of thing commonLuke's *"I Tried to Code a Chess
Engine. It Broke Me."* illustrates in spirit (an engine's edge cases are where the real pain lives,
long after the core search "works"):

- Crashing on an edge case — no legal moves, promotion, en passant — shows up reliably over a few
  hundred games against even a random opponent. Fuzz against `baselines/random` before trusting a
  change.
- Blowing the 90s import budget loading weights or JIT-compiling.
- Writing anywhere outside `/tmp`.
- Running more threads than the one available core — `torch.set_num_threads(1)` if `torch` is used
  at all.
- Naming a file after a stdlib or dependency module (`chess.py`, `random.py`, `types.py`) — the
  zip is first on `sys.path`, so it silently shadows the real module and fails in a way that looks
  unrelated to the actual bug.
- Measuring a change on too few games: per `docs/IDEAS.md`, a 3%-strength change needs *hundreds*
  of games to be visible, so use `make arena` at a fast time control, alternate colours, and always
  compare against your own last version, not just the baselines.

## 7. First-commit strategy

Goal for commit #1: a correct, legal-always, time-safe engine that clearly beats
`baselines/greedy` and gives `baselines/minimax` a real game — the two milestones this repo already
measures itself against. Everything ML-flavored is explicitly deferred; it only pays off once this
foundation is solid and measured.

**Build, in this order:**

1. **Evaluation: material + piece-square tables.** Standard PST values (e.g. the well-known
   simplified evaluation function tables), no mobility or king-safety terms yet. Cheap, fast, and
   gives a stable baseline to diff every later change against.
2. **Search: negamax + alpha-beta**, fail-soft, no move ordering yet — get it producing legal,
   correct results first.
3. **Move ordering: MVV-LVA on captures, TT move first once step 5 exists.** This is where most of
   alpha-beta's real power shows up.
4. **Iterative deepening** driven by a time budget computed from `time_left_ms`, checked every N
   nodes inside the search (not only between `get_move` calls), always keeping the last fully
   completed depth's best move as the fallback return value.
5. **Quiescence search** on captures at the leaves, before trusting the static evaluation.
6. **Transposition table**, keyed by Zobrist hash via `chess.polyglot.zobrist_hash`, stored on a
   module-level dict so it survives across moves within the game; size-capped so it can't grow
   unbounded over a long game inside the 2 GB limit.
7. **Defensive correctness pass**: always fall back to any legal move (`random.choice(list(board.legal_moves))`,
   as `agent.py` already does) if the search is interrupted before finding anything, and explicitly
   test promotions, en passant, and the no-legal-moves-left case.
8. **Repetition awareness**: track visited position hashes so the engine doesn't shuffle away a won
   game via automatic threefold-repetition claims.

**Explicitly deferred past commit #1** (each is a separate, measurable change):
numba-jitted move generation/evaluation, a trained NNUE-style ONNX evaluation network, killer-move
and history heuristics, null-move pruning and LMR, an opening book. Add them one at a time and
verify each with `make arena` against the *previous* committed version before layering the next —
per `docs/IDEAS.md`, "better than my last one" is the only comparison that actually matters here.

**Validate every step with:**
```
make play                 # sanity: one real-time-control game, watch for crashes/flags
make arena                # 20 fast games vs baselines/greedy — the first milestone to clear
make gate                 # ruff + mypy strict + a couple of games that must finish cleanly
uv run python -m harness.arena --opponent ../previous-agent-copy --games 200   # the real comparison once step 1–8 lands
```

## 8. Video-by-video reference

| Video | Creator | What to take from it |
|---|---|---|
| [Why AI Chess Bots Are Virtually Unbeatable](https://www.youtube.com/watch?v=CdFLEfRr3Qk) | WIRED, ft. GothamChess | Framing/motivation for why search depth + evaluation compound into superhuman play; not implementation detail |
| [The Fascinating Programming of a Chess Engine](https://www.youtube.com/watch?v=w4FFX_otR-4) | Bartek Spitza | Gentle intro to board representation (bitboards) and minimax with alpha-beta — the on-ramp before Lague's video |
| [Coding Adventure: Making a Better Chess Bot](https://www.youtube.com/watch?v=_vqlIPDR2TU) | Sebastian Lague | The deep technical one: magic bitboards, Zobrist hashing/transposition tables, killer moves, LMR, king/pawn evaluation, testing against real opposition — closest analogue to this repo's `docs/IDEAS.md` search checklist |
| [I Ran a Chess Programming Tournament!](https://www.youtube.com/watch?v=Ne40a5LkK6A) | Sebastian Lague | A real size/constraint-limited community tournament (his C# Chess-Challenge) — evidence for how much strategy diversity and surprising results (a faithful vs. modernized Turochamp placing very differently) show up even under tight constraints, similar in spirit to this competition's format |
| [I Tried to Code a Chess Engine. It Broke Me.](https://www.youtube.com/watch?v=UqCXwP1F-ho) | commonLuke | Cautionary tale on where the real difficulty lives — not the algorithms, but the edge cases and debugging once the core search "works"; the same lesson as §6 above |
| [chessprogramming.org](https://www.chessprogramming.org/Main_Page) | Chess Programming Wiki | Canonical reference for every term above (search, evaluation, board representation, NNUE, LMR) once a specific technique needs a deeper dive |
