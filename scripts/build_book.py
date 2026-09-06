"""Builds book/book.bin: a small polyglot opening book from named opening theory.

Not part of the runtime path -- agent.py only reads the prebuilt file at import time.
Re-run this whenever the opening lines below change:

    uv run python scripts/build_book.py

Entries come from hand-picked, well-known lines of standard opening theory (not derived
from any specific engine's search or self-play), matching the rules' "opening books ...
are fine" carve-out, distinct from a banned engine-move lookup table.
"""

from __future__ import annotations

import struct
from pathlib import Path

import chess
import chess.polyglot

OUTPUT = Path(__file__).resolve().parent.parent / "book" / "book.bin"

# uci moves from the starting position; weight is this line's relative popularity, used to
# pick among multiple book moves offered at the same position (see agent.py's book probe).
LINES: list[tuple[int, list[str]]] = [
    (10, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]),  # Italian Game
    (10, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),  # Ruy Lopez
    (8, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"]),  # Open Sicilian
    (6, ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"]),  # French Defense
    (6, ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"]),  # Caro-Kann
    (6, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"]),  # Scandinavian Defense
    (9, ["d2d4", "d7d5", "c2c4"]),  # Queen's Gambit
    (7, ["d2d4", "g8f6", "c2c4", "g7g6"]),  # King's Indian setup
    (7, ["c2c4", "e7e5", "b1c3"]),  # English Opening
]

PROMOTION_BITS = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def move_bits(board: chess.Board, move: chess.Move) -> int:
    """Encode `move` in polyglot's 16-bit format (matches chess.polyglot's own decoder)."""
    to_square = move.to_square
    if board.is_castling(move):
        # Polyglot's castling quirk: the king is encoded as capturing its own rook.
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        to_square = chess.square(7 if kingside else 0, chess.square_rank(move.to_square))
    return (
        (PROMOTION_BITS[move.promotion] << 12)
        | (chess.square_rank(move.from_square) << 9)
        | (chess.square_file(move.from_square) << 6)
        | (chess.square_rank(to_square) << 3)
        | chess.square_file(to_square)
    )


def main() -> None:
    entries: dict[int, list[tuple[int, int]]] = {}
    for weight, uci_moves in LINES:
        board = chess.Board()
        for uci in uci_moves:
            move = chess.Move.from_uci(uci)
            key = chess.polyglot.zobrist_hash(board)
            entries.setdefault(key, []).append((move_bits(board, move), weight))
            board.push(move)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as handle:
        for key in sorted(entries):
            for bits, weight in entries[key]:
                handle.write(struct.pack(">QHHI", key, bits, weight, 0))

    total = sum(len(v) for v in entries.values())
    print(f"wrote {OUTPUT} ({total} entries, {OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
