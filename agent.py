"""The submission entrypoint. The platform imports this file and calls get_move."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import chess
import chess.polyglot

# Import time runs once per game, inside a 90 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

MIN_MOVES_REMAINING = 20  # assume at least this many moves left when nothing better is known
MOVE_OVERHEAD_S = 0.3  # margin left on the clock for the watchdog and process overhead

MAX_SEARCH_DEPTH = 64  # a generous safety cap; the deadline is what actually ends the search
MATE_SCORE = 100_000.0

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
        return _evaluate(board)

    best = float("-inf")
    for move in _order_moves(board, legal_moves):
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, deadline)
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


def _order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Captures first, sorted by MVV-LVA; quiet moves keep the order they arrived in.

    Doesn't change what _negamax evaluates, only the order it tries children in -- alpha-beta
    only prunes well when strong moves are seen first, so this is what turns step 2's search
    from full-width into something that actually benefits from the alpha-beta window.
    """
    return sorted(moves, key=lambda move: _capture_score(board, move), reverse=True)


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
