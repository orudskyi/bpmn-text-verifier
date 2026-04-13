"""Module A — Process description text loader.

Reads a plain-text process description and splits it into semantic
:class:`~src.models.TextFragment` objects (one logical clause or action
per fragment).

No external NLP libraries are used — the descriptions are short, well-written
English documents that can be reliably split with regular expressions.

Typical usage::

    from src.module_a.text_loader import load_text

    fragments = load_text("data/dispatch/DispatchDescription.txt")
    for f in fragments:
        print(f.id, f.text)
"""

import logging
import re
from pathlib import Path

from src.models import TextFragment

logger = logging.getLogger(__name__)

# ── Clause-splitting configuration ───────────────────────────────────────────

# Words/phrases that, when following ", " inside a sentence, signal a new
# logical clause worth splitting into a separate fragment.
_CLAUSE_STARTERS = (
    "if",
    "in case",
    "in these cases",
    "in the meantime",
    "therefore",
    "when",
    "and",
    "but",
)

# Built as a single alternation so we only compile once.
_CLAUSE_STARTER_RE = re.compile(
    r",\s+(?=" + "|".join(re.escape(s) for s in _CLAUSE_STARTERS) + r")",
    flags=re.IGNORECASE,
)

# Leading conjunctions to strip from the beginning of a fragment.
_LEADING_CONJUNCTIONS_RE = re.compile(
    r"^(and|but|or|therefore)\s+",
    flags=re.IGNORECASE,
)

# Minimum word count for a fragment to be kept.
_MIN_WORDS = 3


# ── Public API ────────────────────────────────────────────────────────────────


def load_text(file_path: str) -> list[TextFragment]:
    """Load a process description and split it into semantic fragments.

    The splitting strategy (no external NLP libraries):

    1. Read the file as UTF-8 text.
    2. Split on sentence boundaries (``". "`` or end-of-string period).
    3. For each sentence, further split on clause-starting words
       (``if``, ``in case``, ``therefore``, ``when``, ``and``, …)
       — only when both sub-clauses are longer than 5 words.
    4. Clean each fragment: strip whitespace, remove leading conjunctions,
       discard fragments shorter than 3 words.
    5. Assign sequential IDs ``"frag_01"``, ``"frag_02"``, … and preserve
       the original sentence index.

    Args:
        file_path: Path to the ``.txt`` file with the process description.

    Returns:
        List of :class:`~src.models.TextFragment` objects, each representing
        one clause or action.  The list is never empty for a non-trivial input.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    raw = path.read_text(encoding="utf-8")
    logger.info("Loaded text file: %s (%d chars)", path, len(raw))

    sentences = _split_sentences(raw)
    logger.debug("Split into %d sentences", len(sentences))

    fragments: list[TextFragment] = []
    frag_counter = 0

    for sent_idx, sentence in enumerate(sentences):
        clauses = _split_clauses(sentence)
        for clause in clauses:
            cleaned = _clean(clause)
            if cleaned and _word_count(cleaned) >= _MIN_WORDS:
                frag_counter += 1
                frag_id = f"frag_{frag_counter:02d}"
                fragments.append(
                    TextFragment(id=frag_id, text=cleaned, sentence_index=sent_idx)
                )
                logger.debug("  %s [sent %d]: %r", frag_id, sent_idx, cleaned)

    logger.info("Extracted %d fragments from %s", len(fragments), path.name)
    return fragments


# ── Private helpers ───────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split raw text into individual sentences.

    Handles:
    * ``". "`` — the standard sentence boundary inside a paragraph.
    * A trailing period at end-of-string.
    * Windows-style ``\\r\\n`` line endings.

    Args:
        text: Raw text content of the description file.

    Returns:
        Non-empty list of sentence strings (whitespace stripped).
    """
    # Normalise line endings and collapse whitespace across lines.
    normalised = re.sub(r"\s+", " ", text).strip()

    # Split on ". " keeping no delimiter (we discard the period itself).
    parts = re.split(r"\.\s+", normalised)

    sentences = []
    for part in parts:
        # Remove a trailing period that remains after the last sentence.
        cleaned = part.rstrip(". \t")
        if cleaned:
            sentences.append(cleaned)

    return sentences


def _split_clauses(sentence: str) -> list[str]:
    """Further split a sentence on internal clause boundaries.

    A split is performed only when the clause starter follows a comma
    **and** both resulting sub-clauses contain more than 5 words —
    to avoid fragmenting short noun phrases like "in case of small amounts".

    Args:
        sentence: A single cleaned sentence string.

    Returns:
        One or more clause strings.
    """
    parts = _CLAUSE_STARTER_RE.split(sentence)

    if len(parts) == 1:
        return parts  # no split possible

    result: list[str] = []
    for part in parts:
        # Reject fragments too short to stand on their own.
        if _word_count(part) > 5:
            result.append(part)
        elif result:
            # Too short to be independent — merge back into the previous clause.
            result[-1] = result[-1] + ", " + part
        else:
            result.append(part)

    return result if result else [sentence]


def _clean(text: str) -> str:
    """Normalise a single clause fragment.

    * Strips leading/trailing whitespace.
    * Removes leading conjunctions (``and``, ``but``, ``or``, ``therefore``).
    * Collapses internal whitespace sequences to a single space.

    Args:
        text: Raw clause string.

    Returns:
        Cleaned string (may be empty if the input was whitespace-only).
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = _LEADING_CONJUNCTIONS_RE.sub("", text)
    return text.strip()


def _word_count(text: str) -> int:
    """Count whitespace-delimited words in *text*.

    Args:
        text: Any string.

    Returns:
        Integer word count (0 for empty/whitespace-only strings).
    """
    return len(text.split())
