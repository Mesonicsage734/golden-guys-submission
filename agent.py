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

# How many times we've been asked about each position (by Zobrist hash) this game. The
# referee claims threefold repetition automatically, so a search that consults this can
# avoid handing away a won game by shuffling into a draw. Nothing reads it yet -- later
# commits wire it into move selection.
_position_counts: dict[int, int] = {}

TT_SIZE = 1 << 20  # slots; a fixed size bounds memory instead of growing for the whole game
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# One slot per (key & (TT_SIZE - 1)), always overwritten on collision -- simple, and the
# memory cost is bounded by TT_SIZE regardless of how long the game runs (comfortably inside
# the 2 GB budget: at a few hundred bytes a slot, a full table is well under 1 GB). Persists
# across moves within a game (module state), reset fresh for the next one like everything else.
TTEntry = tuple[int, int, float, int, chess.Move]
_transposition_table: list[TTEntry | None] = [None] * TT_SIZE


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
    """
    board = chess.Board(fen)
    _remember(board)
    try:
        return _choose_move(board, Deadline.from_time_left(time_left_ms))
    except Exception:
        return random.choice(list(board.legal_moves)).uci()


def _remember(board: chess.Board) -> None:
    key = chess.polyglot.zobrist_hash(board)
    _position_counts[key] = _position_counts.get(key, 0) + 1


def _choose_move(board: chess.Board, deadline: Deadline) -> str:
    """Iterative deepening: search depth 1, then 2, then 3..., always keeping the last
    fully completed pass's best move ready to return the instant the deadline hits.

    A root move is scored by searching several plies past it, never by evaluating the
    resulting position directly -- _evaluate only ever runs at the leaves _negamax reaches.
    An incomplete pass is discarded entirely rather than trusted: _search_root raises
    SearchTimeout without returning if the budget runs out partway through it, so
    `best_move` here only ever gets updated from a pass that scored every root move.
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
    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        try:
            best_move, _ = _search_root(board, legal_moves, depth, deadline)
        except SearchTimeout:
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
_last_move_from: dict[int, chess.Move] = {}


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

    key = chess.polyglot.zobrist_hash(board)
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
    board: chess.Board, moves: list[chess.Move], depth: int, deadline: Deadline
) -> tuple[chess.Move, float]:
    """Score every move in `moves` by negamax to `depth` plies and return the best.

    Raises SearchTimeout (via deadline.check(), possibly from deep inside _negamax) the
    moment the budget runs out, without returning -- the caller relies on this to discard
    an incomplete pass rather than compare a partial subset of moves.
    """
    best_move = moves[0]
    best_score = float("-inf")
    for move in moves:
        deadline.check()
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, float("-inf"), float("inf"), deadline)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move, best_score


def _move_to_front(moves: list[chess.Move], move: chess.Move) -> list[chess.Move]:
    """Put `move` first for the next pass, keeping the rest in their existing order."""
    return [move, *(candidate for candidate in moves if candidate != move)]


def _negamax(
    board: chess.Board, depth: int, alpha: float, beta: float, deadline: Deadline
) -> float:
    """Return the score of `board` for the side to move, `depth` plies from here.

    Fail-soft alpha-beta: `alpha`/`beta` are the caller's window in the side-to-move's own
    sign convention (negamax), and a move that pushes the score at or past `beta` cuts the
    rest of this node's siblings, since the opponent already has a better option elsewhere.
    """
    deadline.check()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_check():
            return -(MATE_SCORE + depth)  # fewer plies remaining here == a faster mate
        return 0.0  # stalemate
    if depth <= 0:
        return _quiescence(board, alpha, beta, deadline, 0)

    key = chess.polyglot.zobrist_hash(board)
    slot = _transposition_table[key & (TT_SIZE - 1)]
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

    original_alpha = alpha
    best = float("-inf")
    best_move = legal_moves[0]
    for move in _order_moves(board, legal_moves, tt_move):
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, deadline)
        finally:
            board.pop()
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    flag = TT_UPPER if best <= original_alpha else TT_LOWER if best >= beta else TT_EXACT
    _transposition_table[key & (TT_SIZE - 1)] = (key, depth, best, flag, best_move)
    return best


MAX_QUIESCENCE_PLY = 32  # a generous safety cap against runaway check-evasion recursion


def _quiescence(
    board: chess.Board, alpha: float, beta: float, deadline: Deadline, ply: int
) -> float:
    """Extend search through captures (and check evasions) past the horizon, so _negamax
    never trusts a static evaluation in the middle of an exchange.

    Not in check: standing pat is sound (a losing capture is never forced), so the static
    eval is a lower bound and only captures are searched further. In check: standing pat
    is unsound -- every legal reply is searched instead, exactly like _negamax's own
    terminal handling, since a position can't be judged "quiet" while it's under attack.
    """
    deadline.check()
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

    for move in _order_moves(board, candidates):
        board.push(move)
        try:
            score = -_quiescence(board, -beta, -alpha, deadline, ply + 1)
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


CAPTURE_ORDER_SCALE = 10  # keeps victim value dominant over attacker value in the sort key


def _order_moves(
    board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None = None
) -> list[chess.Move]:
    """Captures first, sorted by MVV-LVA; quiet moves keep the order they arrived in.
    `tt_move`, if given, goes first of all -- it's the strongest ordering hint available,
    coming from a previous search of this exact position.

    Doesn't change what _negamax evaluates, only the order it tries children in -- alpha-beta
    only prunes well when strong moves are seen first, so this is what turns step 2's search
    from full-width into something that actually benefits from the alpha-beta window.
    """

    def score(move: chess.Move) -> float:
        return float("inf") if move == tt_move else _capture_score(board, move)

    return sorted(moves, key=score, reverse=True)


def _capture_score(board: chess.Board, move: chess.Move) -> float:
    """Most valuable victim, least valuable attacker. -1 for a quiet move (sorts last)."""
    if not board.is_capture(move):
        return -1.0
    victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    victim_value = PIECE_VALUES[victim] if victim is not None else 0.0
    attacker_value = PIECE_VALUES[attacker] if attacker is not None else 0.0
    return CAPTURE_ORDER_SCALE * victim_value - attacker_value


def _evaluate(board: chess.Board) -> float:
    """Material plus piece-square tables, scored for the side to move (negamax convention)."""
    score = _material_and_pst(board)
    return score if board.turn == chess.WHITE else -score


def _material_and_pst(board: chess.Board) -> float:
    """Material plus piece-square tables, from White's perspective."""
    score = 0.0
    for piece_type, table in PIECE_SQUARE_TABLES.items():
        value = PIECE_VALUES[piece_type]
        for square in board.pieces(piece_type, chess.WHITE):
            score += value + table[square] / 100.0
        for square in board.pieces(piece_type, chess.BLACK):
            score -= value + table[chess.square_mirror(square)] / 100.0
    return score
