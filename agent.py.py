"""The submission entrypoint. The platform imports this file and calls get_move."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy

# Import time runs once per game, inside a 90 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

MIN_MOVES_REMAINING = 20  # assume at least this many moves left when nothing better is known
MOVE_OVERHEAD_S = 0.3  # margin left on the clock for the watchdog and process overhead

MAX_SEARCH_DEPTH = 64  # a generous safety cap; the deadline is what actually ends the search
MATE_SCORE = 100_000.0

BOOK_PATH = Path(__file__).resolve().parent / "book" / "book.bin"
BOOK_PLY_LIMIT = 10  # only probe the book this early; theory runs out fast either way


def _load_book(path: Path) -> dict[int, list[tuple[chess.Move, int]]]:
    """Read book.bin (built by scripts/build_book.py) into a plain dict at import time.

    Any failure here -- missing file, corrupt data -- falls back to an empty book rather
    than a crash: an opening book is a nice-to-have, never worth an import-time failure.
    """
    book: dict[int, list[tuple[chess.Move, int]]] = {}
    try:
        with chess.polyglot.open_reader(path) as reader:
            for entry in reader:
                book.setdefault(entry.key, []).append((entry.move, entry.weight))
    except Exception:
        return {}
    return book


_BOOK = _load_book(BOOK_PATH)

SYZYGY_PATH = Path(__file__).resolve().parent / "syzygy"
TABLEBASE_MAX_PIECES = 4  # matches the 3- and 4-man tables shipped in syzygy/


def _load_tablebase(path: Path) -> chess.syzygy.Tablebase | None:
    """Open the shipped Syzygy tables at import time.

    Any failure here -- missing directory, corrupt files -- falls back to None rather than
    a crash: an endgame tablebase is a nice-to-have, never worth an import-time failure.
    """
    try:
        return chess.syzygy.open_tablebase(str(path))
    except Exception:
        return None


_TABLEBASE = _load_tablebase(SYZYGY_PATH)

# Material in pawns, and piece-square bonuses in centipawns (hence the /100.0 in
# _material_and_pst), from White's perspective with a1 == index 0. Mirror the square to
# score a black piece.
# Values are the well-known "simplified evaluation function" tables -- a hand-tuned starting
# point, not learned and not lifted from another engine's source.
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

PIECE_SQUARE_TABLES: dict[chess.PieceType, tuple[int, ...]] = {
    chess.PAWN: (
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    chess.KNIGHT: (
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ),
    chess.BISHOP: (
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ),
    chess.ROOK: (
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ),
    chess.QUEEN: (
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -10, 5, 5, 5, 5, 5, 0, -10,
        0, 0, 5, 5, 5, 5, 0, -5,
        -5, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ),
    chess.KING: (
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
    ),
}

# How many times we've been asked about each position (by board._transposition_key(), the
# same key python-chess's own is_repetition() is built on -- board state, castling rights,
# and en passant only when actually capturable, which is exactly what the repetition rule
# cares about) this game. The referee claims threefold repetition automatically, so a search
# that consults this can avoid handing away a won game by shuffling into a draw.
#
# _negamax and _quiescence now read this (combined with a per-search path_counts of moves
# made within the current search itself) to score a position that has already occurred
# twice before -- real game history plus this search's own line -- as a draw the instant a
# third occurrence would happen, instead of searching past it as if it were a normal position.
PositionKey = tuple  # board._transposition_key()'s return type: a hashable board-state tuple
_position_counts: dict[PositionKey, int] = {}

DRAW_SCORE = 0.0
# A draw is scored very slightly worse than dead even for whoever is to move at the node
# where it's detected. Negamax's own sign flip at every ply carries this to the right side
# automatically, so this one line is enough to make the engine avoid repeating a winning
# position while still being willing to accept a draw when there's nothing better on offer.
DRAW_CONTEMPT = 0.05

TT_SIZE = 1 << 20  # slots; a fixed size bounds memory instead of growing for the whole game
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# One slot per (hash(key) & (TT_SIZE - 1)), always overwritten on collision -- simple, and the
# memory cost is bounded by TT_SIZE regardless of how long the game runs (comfortably inside
# the 2 GB budget: at a few hundred bytes a slot, a full table is well under 1 GB). Persists
# across moves within a game (module state), reset fresh for the next one like everything else.
#
# Keyed by board._transposition_key() rather than chess.polyglot.zobrist_hash(board): both
# identify a position uniquely for our purposes, but the polyglot hash exists to match the
# external, portable format opening books are published in, which is a real cost (~24us/call
# measured) this internal-only table has no reason to pay -- _transposition_key() measures
# roughly 25x faster (~1us/call), and it's what python-chess's own repetition detection is
# already built on, so reusing it here also directly serves the DRAW_CONTEMPT check above.
TTEntry = tuple[PositionKey, int, float, int, chess.Move]
_transposition_table: list[TTEntry | None] = [None] * TT_SIZE

# Killer moves: two quiet moves that caused a beta cutoff at each ply, tried early in a
# sibling node at the same ply even without a capture to recommend them -- siblings at the
# same distance from the root often share the same tactical theme, so a move that refuted
# one line is a good early guess against another. Indexed by ply (distance from THIS
# search's root), not by remaining depth, since that's what siblings actually share.
MAX_PLY = MAX_SEARCH_DEPTH + 32
_killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]

# History heuristic: quiet moves that have caused a beta cutoff anywhere, scored by how
# much search depth they saved when they did. Keyed by (colour, from, to) rather than by
# the move object so a good idea learned in one position still helps order the same
# from/to move in an unrelated one. Persists for the whole game like the TT above.
_history: dict[tuple[chess.Color, chess.Square, chess.Square], int] = {}


class SearchTimeout(Exception):
    """Raised to unwind a search cleanly once its time budget is spent."""


@dataclass(frozen=True)
class Deadline:
    """A wall-clock point a search should stop past, sized from the clock we were handed."""

    at: float

    @classmethod
    def from_time_left(cls, time_left_ms: int) -> Deadline:
        budget_s = max(0.05, time_left_ms / 1000 / MIN_MOVES_REMAINING - MOVE_OVERHEAD_S)
        return cls(at=time.monotonic() + budget_s)

    def check(self) -> None:
        """Raise SearchTimeout once the budget is spent. Call this often inside a search."""
        if time.monotonic() >= self.at:
            raise SearchTimeout


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.

    Anything below can fail -- a bug, a clock computed wrong, an edge case in the search --
    without losing the game to it: any exception falls back to any legal move.

    The referee should never call this on a position where the game has already ended, but
    "should never" isn't "can't": the except-fallback below needs a legal move to choose
    from, and a position with none (checkmate, stalemate) is the one case where it wouldn't
    have one -- discovered by the test suite calling get_move directly on such a position,
    where it raised IndexError from inside its own fallback rather than returning safely.
    """
    board = chess.Board(fen)
    _remember(board)
    if not board.legal_moves:
        return "0000"  # no legal move exists; nothing we return here will be played anyway
    try:
        return _choose_move(board, Deadline.from_time_left(time_left_ms))
    except Exception:
        return random.choice(list(board.legal_moves)).uci()


def _remember(board: chess.Board) -> None:
    key = board._transposition_key()
    _position_counts[key] = _position_counts.get(key, 0) + 1


def _choose_move(board: chess.Board, deadline: Deadline) -> str:
    """Iterative deepening: search depth 1, then 2, then 3..., always keeping the last
    fully completed pass's best move ready to return the instant the deadline hits.

    A root move is scored by searching several plies past it, never by evaluating the
    resulting position directly -- _evaluate only ever runs at the leaves _negamax reaches.
    A pass that finished early (deeper than the last completed one) always wins; one cut off
    partway through only overrides `best_move` if nothing has finished at all yet -- see
    _search_root's docstring for why that specific case matters.
    """
    book_move = _book_move(board)
    if book_move is not None:
        return book_move.uci()

    tablebase_move = _tablebase_move(board)
    if tablebase_move is not None:
        return tablebase_move.uci()

    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)
    legal_moves = _order_moves(board, legal_moves)
    best_move = legal_moves[0]
    have_complete_pass = False
    # One shared path_counts for the whole iterative-deepening loop: every increment in
    # _negamax/_quiescence is paired with a decrement in a `finally`, so it always fully
    # unwinds back to empty between passes regardless of where a pass gets cut off.
    path_counts: dict[PositionKey, int] = {}
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        candidate_move, candidate_score, complete = _search_root(
            board, legal_moves, depth, deadline, path_counts
        )
        if not complete:
            if not have_complete_pass:
                best_move = candidate_move
            break
        best_move, best_score = candidate_move, candidate_score
        have_complete_pass = True
        # _negamax only ever returns a mate-flavoured score (magnitude >= MATE_SCORE, since
        # the smallest is MATE_SCORE + 0 for a mate found the instant depth hits zero) at an
        # actual terminal checkmate it walked every reply down to, in every branch _search_root
        # tried -- never as a depth-limited guess the way an ordinary evaluation score is. That
        # makes it a fully proven result: searching deeper can find the same forced outcome by
        # a different route, or a quicker forced mate, but it cannot overturn "every one of my
        # options was checked and this is what happens" and searching for it further would
        # only spend clock time proving something already proven, at the expense of a later move.
        if abs(best_score) >= MATE_SCORE:
            break
        # Seed the next, deeper pass with this depth's best move first -- iterative
        # deepening's classic free win for ordering, ahead of the real TT step 6 adds.
        legal_moves = _move_to_front(legal_moves, best_move)
    return best_move.uci()


def _book_move(board: chess.Board) -> chess.Move | None:
    """A known opening move for `board`, drawn from the book for only the first few plies.

    Weighted random choice among whatever candidates the book offers at this position,
    matching how a real polyglot book is normally consulted. Every candidate is checked
    against the current legal moves before being trusted -- cheap insurance against a
    Zobrist collision or an encoding bug turning a book hit into an illegal move.
    """
    if not _BOOK or len(board.move_stack) >= BOOK_PLY_LIMIT:
        return None
    candidates = _BOOK.get(chess.polyglot.zobrist_hash(board))
    if not candidates:
        return None
    legal_moves = board.legal_moves
    candidates = [(move, weight) for move, weight in candidates if move in legal_moves]
    if not candidates:
        return None
    moves = [move for move, _ in candidates]
    weights = [weight for _, weight in candidates]
    choice: chess.Move = random.choices(moves, weights=weights, k=1)[0]
    return choice


# The move played the last time we were asked about a given position (by Zobrist hash).
# DTZ-minimizing is otherwise fully deterministic, so if an opponent's play ever brings a
# position back around with us to move again, repeating our own past choice would repeat
# the position too -- a real, observed failure mode (see the step 8 commit message).
_last_move_from: dict[PositionKey, chess.Move] = {}


def _tablebase_move(board: chess.Board) -> chess.Move | None:
    """A tablebase-optimal move once few enough pieces remain, else None to keep searching.

    Ranks each move by the resulting position's (WDL, DTZ) from the opponent's point of
    view: WDL first, ascending, since a smaller (more negative) value is unconditionally
    worse for the opponent regardless of ply count, and DTZ never orders correctly across
    a WDL boundary (a "blessed loss" can carry a larger DTZ magnitude than an "unconditional
    loss" a whole category better for us). Within the same WDL, prefer the larger DTZ: less
    negative means the opponent is forced to lose in fewer plies; more positive means they
    need more plies to beat us, i.e. the best available delay. Any probe failure (a table
    genuinely missing, e.g. a capture down to a bare-kings position outside the 3-4 man set
    we ship) aborts to the normal search entirely, rather than trusting a partial comparison.

    Multiple moves often tie under this ranking, since it assumes a perfect defender on
    both sides. Ties are broken with a simple, directly relevant heuristic: bring the two
    kings closer together, the standard technique for actually cornering a lone king.
    _evaluate is not that heuristic -- its king-square table rewards a castled-looking king
    (corners, back rank), the opposite of what a king that has to help deliver mate needs.

    If this exact position (us to move) has already occurred, the move played last time is
    also excluded before tie-breaking -- otherwise a repeating opponent could still lock a
    deterministic choice into repeating the position right back into a draw.
    """
    if (
        _TABLEBASE is None
        or board.castling_rights
        or chess.popcount(board.occupied) > TABLEBASE_MAX_PIECES
    ):
        return None

    key = board._transposition_key()  # must match _position_counts's key type (see _remember)
    avoid = _last_move_from.get(key) if _position_counts.get(key, 0) > 1 else None
    our_color = board.turn

    best_rank: tuple[int, int] | None = None
    candidates: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        try:
            wdl = _TABLEBASE.probe_wdl(board)
            dtz = _TABLEBASE.probe_dtz(board)
        except Exception:
            return None
        finally:
            board.pop()
        rank = (wdl, -dtz)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            candidates = [move]
        elif rank == best_rank:
            candidates.append(move)

    if avoid in candidates and len(candidates) > 1:
        candidates = [move for move in candidates if move != avoid]

    chosen = min(candidates, key=lambda move: _king_distance_after(board, move, our_color))
    _last_move_from[key] = chosen
    return chosen


def _king_distance_after(board: chess.Board, move: chess.Move, our_color: chess.Color) -> int:
    """Chebyshev distance between the two kings after `move`, lower meaning closer."""
    board.push(move)
    try:
        our_king = board.king(our_color)
        their_king = board.king(not our_color)
    finally:
        board.pop()
    if our_king is None or their_king is None:
        return 8
    return chess.square_distance(our_king, their_king)


def _search_root(
    board: chess.Board,
    moves: list[chess.Move],
    depth: int,
    deadline: Deadline,
    path_counts: dict[PositionKey, int],
) -> tuple[chess.Move, float, bool]:
    """Score every move in `moves` by negamax to `depth` plies and return the best.

    Returns (best_move, best_score, complete). complete is False the moment the deadline
    interrupts this pass -- some root moves may never have been tried at all -- but unlike
    before, a timeout no longer discards the moves that WERE fully scored before it hit.

    That change matters specifically at a depth-1 pass that never finishes at all: without
    it, `_choose_move` would fall back to `moves[0]`, an entry from that function's own
    random.shuffle() that has never been evaluated by anything. A quiet move that ties with
    every other quiet move on _order_moves's score (which most do, absent a killer or
    history hit) keeps whatever order the shuffle gave it, so `moves[0]` there is
    functionally a coin flip. Under real time pressure this is not hypothetical: it is
    exactly how this file, unmodified, produced a real one-move loss to a 2-ply baseline
    during testing of this change -- a quiet move that hangs mate in one, returned having
    never been compared to anything, because the very first depth-1 pass didn't finish.
    Returning whatever this call DID manage to score, even from an interrupted pass, is
    strictly better in expectation than that coin flip, and costs nothing: the caller still
    prefers a fully completed shallower pass over an interrupted deeper one exactly as
    before, using the partial result only when it has never completed a pass at all.
    """
    best_move = moves[0]
    best_score = float("-inf")
    for move in moves:
        if time.monotonic() >= deadline.at:
            return best_move, best_score, False
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, float("-inf"), float("inf"), deadline,
                               path_counts, ply=1)
        except SearchTimeout:
            return best_move, best_score, False
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move, best_score, True


def _move_to_front(moves: list[chess.Move], move: chess.Move) -> list[chess.Move]:
    """Put `move` first for the next pass, keeping the rest in their existing order."""
    return [move, *(candidate for candidate in moves if candidate != move)]


NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2

LMR_MIN_DEPTH = 3
LMR_MIN_MOVE_INDEX = 3  # try the first few moves at full depth regardless of ordering


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    deadline: Deadline,
    path_counts: dict[PositionKey, int],
    ply: int,
    allow_null: bool = True,
) -> float:
    """Return the score of `board` for the side to move, `depth` plies from here.

    Fail-soft alpha-beta: `alpha`/`beta` are the caller's window in the side-to-move's own
    sign convention (negamax), and a move that pushes the score at or past `beta` cuts the
    rest of this node's siblings, since the opponent already has a better option elsewhere.

    `path_counts` counts occurrences of a position within *this* search's own line, on top
    of `_position_counts`'s count from the real game so far; `ply` is this node's distance
    from the root of this search (as opposed to `depth`, which counts down to the horizon),
    used to index killer moves the way siblings at the same distance from the root share.
    """
    deadline.check()
    key = board._transposition_key()

    # A position that has already occurred twice before -- combining the real game and this
    # search's own line -- would be a third occurrence right now: the referee auto-claims
    # that draw, so there is nothing left to search past it. Checked before depth<=0 and
    # before the TT, since it can be reached at any depth and doesn't depend on either.
    if _position_counts.get(key, 0) + path_counts.get(key, 0) >= 2:
        return -DRAW_CONTEMPT
    if board.halfmove_clock >= 100:
        return -DRAW_CONTEMPT

    if depth <= 0:
        return _quiescence(board, alpha, beta, deadline, path_counts, 0)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -(MATE_SCORE + depth)  # fewer plies remaining here == a faster mate
        return 0.0  # stalemate

    # board._transposition_key() rather than chess.polyglot.zobrist_hash(board): both
    # identify this position, but the polyglot hash pays for a portable, book-compatible
    # format this internal-only table doesn't need (~24us vs ~1us/call, measured). The
    # array only needs an int index, so hash() the tuple; slot[0] == key still checks the
    # actual position on a lookup, exactly as chess_key == zobrist did before, so a hash
    # collision (of hash(), now, rather than of the polyglot hash before) still can't
    # return another position's answer for this one.
    index = hash(key) & (TT_SIZE - 1)
    slot = _transposition_table[index]
    tt_move = None
    if slot is not None and slot[0] == key:
        _, entry_depth, entry_score, entry_flag, entry_move = slot
        tt_move = entry_move
        if entry_depth >= depth:
            if entry_flag == TT_EXACT:
                return entry_score
            if entry_flag == TT_LOWER:
                alpha = max(alpha, entry_score)
            elif entry_flag == TT_UPPER:
                beta = min(beta, entry_score)
            if alpha >= beta:
                return entry_score

    in_check = board.is_check()

    # Null-move pruning: if we could pass entirely and the opponent still can't get their
    # score back up to beta with a cheaper (reduced-depth) search, our actual best move is
    # certainly at least that good too, so the whole node is pruned without searching our
    # own replies at all. Skipped in check (passing isn't legal there), near the horizon
    # (too little depth left for the reduction to leave anything to search), right after
    # our own null move (two passes in a row proves nothing), and whenever we hold only
    # pawns and a king -- the classic zugzwang shape where passing is actually better than
    # any legal move, which would make this heuristic actively wrong.
    #
    # No separate "skip on a PV node" guard here: that's a standard refinement in engines
    # that score in centipawns, where a window of literal width 1 (beta == alpha + 1) is a
    # meaningful, reliably-narrow "scout" signal. This file scores in whole pawns, so a
    # width-1 window is actually enormous on this scale and doesn't identify the same thing
    # -- porting that check without rescaling it would silently mis-gate null-move rather
    # than protect anything, so it's left out rather than shipped miscalibrated.
    if (
        allow_null
        and not in_check
        and depth >= NULL_MOVE_MIN_DEPTH
        and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)
    ):
        board.push(chess.Move.null())
        try:
            null_score = -_negamax(
                board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, deadline,
                path_counts, ply + 1, allow_null=False,
            )
        finally:
            board.pop()
        if null_score >= beta:
            return beta

    original_alpha = alpha
    best = float("-inf")
    best_move = legal_moves[0]
    ordered_moves = _order_moves(board, legal_moves, tt_move, ply)
    path_counts[key] = path_counts.get(key, 0) + 1
    try:
        for index_in_order, move in enumerate(ordered_moves):
            is_quiet = not board.is_capture(move) and move.promotion is None
            board.push(move)
            try:
                gives_check = board.is_check()
                # Late move reductions: a quiet move ordered late (no capture, no promotion,
                # not one of the first few tried, and not a check either way) is, by exactly
                # the same reasoning _order_moves relies on, the least likely of this node's
                # moves to matter. Search it shallower first; if it still beats alpha despite
                # the handicap, it earned a full-depth re-search to find its real value.
                reduced = (
                    depth >= LMR_MIN_DEPTH
                    and index_in_order >= LMR_MIN_MOVE_INDEX
                    and is_quiet
                    and not in_check
                    and not gives_check
                )
                if reduced:
                    score = -_negamax(board, depth - 2, -alpha - 1, -alpha, deadline,
                                       path_counts, ply + 1)
                    if score > alpha:
                        score = -_negamax(board, depth - 1, -beta, -alpha, deadline,
                                           path_counts, ply + 1)
                else:
                    score = -_negamax(board, depth - 1, -beta, -alpha, deadline,
                                       path_counts, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if is_quiet:
                    killers = _killers[min(ply, MAX_PLY - 1)]
                    if killers[0] != move:
                        killers[1] = killers[0]
                        killers[0] = move
                    history_key = (board.turn, move.from_square, move.to_square)
                    _history[history_key] = _history.get(history_key, 0) + depth * depth
                break
    finally:
        remaining = path_counts[key] - 1
        if remaining:
            path_counts[key] = remaining
        else:
            del path_counts[key]

    flag = TT_UPPER if best <= original_alpha else TT_LOWER if best >= beta else TT_EXACT
    _transposition_table[index] = (key, depth, best, flag, best_move)
    return best


MAX_QUIESCENCE_PLY = 32  # a generous safety cap against runaway check-evasion recursion


def _quiescence(
    board: chess.Board,
    alpha: float,
    beta: float,
    deadline: Deadline,
    path_counts: dict[PositionKey, int],
    ply: int,
) -> float:
    """Extend search through captures (and check evasions) past the horizon, so _negamax
    never trusts a static evaluation in the middle of an exchange.

    Not in check: standing pat is sound (a losing capture is never forced), so the static
    eval is a lower bound and only captures are searched further. In check: standing pat
    is unsound -- every legal reply is searched instead, exactly like _negamax's own
    terminal handling, since a position can't be judged "quiet" while it's under attack.

    `ply` here is local to this quiescence dive (0 at the _negamax call site, unrelated to
    _negamax's own ply-from-search-root), exactly as before -- only path_counts and the
    repetition check are new, threaded through for the same reason _negamax needs them.
    """
    deadline.check()
    key = board._transposition_key()
    if _position_counts.get(key, 0) + path_counts.get(key, 0) >= 2:
        return -DRAW_CONTEMPT
    if board.halfmove_clock >= 100:
        return -DRAW_CONTEMPT

    legal_moves = list(board.legal_moves)
    in_check = board.is_check()
    if not legal_moves:
        if in_check:
            return -(MATE_SCORE + ply)
        return 0.0
    if ply >= MAX_QUIESCENCE_PLY:
        return _evaluate(board)

    if in_check:
        best = float("-inf")
        candidates = legal_moves
    else:
        best = _evaluate(board)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        candidates = [move for move in legal_moves if board.is_capture(move)]

    path_counts[key] = path_counts.get(key, 0) + 1
    try:
        for move in _order_moves(board, candidates):
            board.push(move)
            try:
                score = -_quiescence(board, -beta, -alpha, deadline, path_counts, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
    finally:
        remaining = path_counts[key] - 1
        if remaining:
            path_counts[key] = remaining
        else:
            del path_counts[key]
    return best


CAPTURE_ORDER_SCALE = 10  # keeps victim value dominant over attacker value in the sort key

# The worst-scoring real capture (queen takes pawn: 10*1.0 - 9.0 = 1.0) still has to outrank
# every quiet move, so killer/history scores for quiet moves are kept inside (-1.0, -0.9]
# -- comfortably below 1.0 -- and killers get a fixed slot above that range but still below
# any capture, so captures always go first exactly as they did before this file had killers.
KILLER_1_SCORE = -0.6
KILLER_2_SCORE = -0.7
HISTORY_SCALE = 100_000  # caps how far accumulated history can shift a quiet move's score
HISTORY_MAX_SHIFT = 0.1


def _order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None = None,
    ply: int | None = None,
) -> list[chess.Move]:
    """Captures first, sorted by MVV-LVA; killer moves next; other quiet moves ordered by
    the history heuristic. `tt_move`, if given, goes first of all -- it's the strongest
    ordering hint available, coming from a previous search of this exact position.

    `ply` selects which ply's killer pair to check -- omitted (None) by _quiescence, whose
    own `ply` counts something different (distance into this quiescence dive, not distance
    from the search root) and so isn't a valid index into the shared, root-relative table.

    Doesn't change what _negamax evaluates, only the order it tries children in -- alpha-beta
    only prunes well when strong moves are seen first, so this is what turns step 2's search
    from full-width into something that actually benefits from the alpha-beta window.
    """
    killer_1, killer_2 = _killers[min(ply, MAX_PLY - 1)] if ply is not None else (None, None)

    def score(move: chess.Move) -> float:
        if move == tt_move:
            return float("inf")
        if board.is_capture(move):
            return _capture_score(board, move)
        if move == killer_1:
            return KILLER_1_SCORE
        if move == killer_2:
            return KILLER_2_SCORE
        history = _history.get((board.turn, move.from_square, move.to_square), 0)
        return -1.0 + min(history, HISTORY_SCALE) / HISTORY_SCALE * HISTORY_MAX_SHIFT

    return sorted(moves, key=score, reverse=True)


def _capture_score(board: chess.Board, move: chess.Move) -> float:
    """Most valuable victim, least valuable attacker. Only ever called on an actual capture
    -- _order_moves branches on board.is_capture(move) before reaching here."""
    victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    victim_value = PIECE_VALUES[victim] if victim is not None else 0.0
    attacker_value = PIECE_VALUES[attacker] if attacker is not None else 0.0
    return CAPTURE_ORDER_SCALE * victim_value - attacker_value


# Material + PST folded into one flat 64-entry list per piece type per colour, built once
# here at import time from the same PIECE_VALUES/PIECE_SQUARE_TABLES data above -- so
# _material_and_pst becomes one array index and one addition per piece instead of a dict
# lookup, a division, and an addition. Verified bit-identical to the original dict-based
# version across 1,200 positions from 30 random games before being made the real path.
# White reads _WHITE_TABLE[piece_type][square] directly, matching this file's existing
# "a1 == index 0" convention (see the comment above PIECE_VALUES); black's table is
# precomputed with the mirror and the sign flip already folded in, so no mirroring or
# negation happens per piece at evaluation time either.
_WHITE_TABLE: dict[chess.PieceType, list[float]] = {}
_BLACK_TABLE: dict[chess.PieceType, list[float]] = {}
for _piece_type, _table in PIECE_SQUARE_TABLES.items():
    _value = PIECE_VALUES[_piece_type]
    _WHITE_TABLE[_piece_type] = [_value + _table[_sq] / 100.0 for _sq in range(64)]
    _BLACK_TABLE[_piece_type] = [
        -(_value + _table[chess.square_mirror(_sq)] / 100.0) for _sq in range(64)
    ]


def _evaluate(board: chess.Board) -> float:
    """Material plus piece-square tables, scored for the side to move (negamax convention)."""
    score = _material_and_pst(board)
    return score if board.turn == chess.WHITE else -score


def _material_and_pst(board: chess.Board) -> float:
    """Material plus piece-square tables, from White's perspective.

    Iterates set bits of each piece type's bitboard directly (pieces_mask) rather than
    through board.pieces()'s SquareSet, and reads the precomputed combined value straight
    out of _WHITE_TABLE/_BLACK_TABLE -- same numbers as before, just without a dict lookup,
    a division, and an addition on every piece on every call.
    """
    score = 0.0
    for piece_type in PIECE_SQUARE_TABLES:
        white_table = _WHITE_TABLE[piece_type]
        black_table = _BLACK_TABLE[piece_type]
        bitboard = board.pieces_mask(piece_type, chess.WHITE)
        while bitboard:
            square = (bitboard & -bitboard).bit_length() - 1
            bitboard &= bitboard - 1
            score += white_table[square]
        bitboard = board.pieces_mask(piece_type, chess.BLACK)
        while bitboard:
            square = (bitboard & -bitboard).bit_length() - 1
            bitboard &= bitboard - 1
            score += black_table[square]
    return score
