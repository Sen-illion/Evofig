"""Registration helper for the deck-recovery maintenance command."""

from __future__ import annotations


def register_deck_adoption_command(subparsers) -> None:
    parser = subparsers.add_parser("adopt-undelivered-decks", help="Re-adjudicate decks an earlier batch drew but did not deliver, and copy them into a batch root without calling a provider.")
    parser.add_argument("--from", dest="sources", action="append", required=True, help="Batch root that may hold the earlier attempt. Repeatable; the newest readable attempt wins.")
    parser.add_argument("--out", required=True, help="Batch root to adopt into: each paper lands in runs/<paper>/attempt_01.")
    parser.add_argument("--subset", required=True, help="Corpus subset version whose ledger names the papers to adopt.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
