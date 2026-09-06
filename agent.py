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
    """Placeholder search: step 2 replaces the body with real negamax + evaluation.

    The shape survives every later commit: walk the candidates, check the deadline before
    each one, and keep the best-so-far ready to return the instant time runs out.
    """
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)
    best = legal_moves[0]
    try:
        for candidate in legal_moves:
            deadline.check()
            best = candidate
    except SearchTimeout:
        pass
    return best.uci()
