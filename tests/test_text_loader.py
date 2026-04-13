"""Tests for Module A — text loader (TASK_02).

Uses the two provided sample process descriptions as fixtures:
  - data/dispatch/DispatchDescription.txt
  - data/recourse/RecourseDesription.txt   (note: typo in filename is intentional)
"""

import pytest

from src.module_a.text_loader import load_text

# ── File paths ────────────────────────────────────────────────────────────────

DISPATCH_FILE = "data/dispatch/DispatchDescription.txt"
RECOURSE_FILE = "data/recourse/RecourseDesription.txt"


# ── Shared invariants ─────────────────────────────────────────────────────────


def _assert_invariants(fragments: list) -> None:
    """Check properties that must hold for any valid fragment list."""
    assert len(fragments) > 0, "Fragment list must not be empty"
    for i, f in enumerate(fragments):
        assert f.text.strip() != "", f"Fragment {f.id} has empty text"
        assert f.id == f"frag_{i + 1:02d}", f"Fragment IDs must be sequential; got {f.id!r}"
        assert len(f.text.split()) >= 3, f"Fragment {f.id!r} is too short: {f.text!r}"
        assert "\n" not in f.text, f"Fragment {f.id!r} contains a newline"
        assert "\r" not in f.text, f"Fragment {f.id!r} contains a carriage return"


# ── Dispatch tests ────────────────────────────────────────────────────────────


def test_dispatch_minimum_fragment_count():
    """Dispatch description must yield at least 7 meaningful fragments."""
    fragments = load_text(DISPATCH_FILE)
    assert len(fragments) >= 7, f"Got only {len(fragments)} fragments"


def test_dispatch_invariants():
    """All dispatch fragments must satisfy the shared invariants."""
    _assert_invariants(load_text(DISPATCH_FILE))


def test_dispatch_contains_key_concepts():
    """Key domain concepts must appear somewhere in the fragments."""
    fragments = load_text(DISPATCH_FILE)
    all_text = " ".join(f.text for f in fragments).lower()
    assert "secretary" in all_text
    assert "package" in all_text
    assert "logistic" in all_text


def test_dispatch_sentence_indices_non_negative():
    """sentence_index values must be zero-based non-negative integers."""
    fragments = load_text(DISPATCH_FILE)
    for f in fragments:
        assert f.sentence_index >= 0


def test_dispatch_sentence_indices_ordered():
    """sentence_index values must be non-decreasing across fragments."""
    fragments = load_text(DISPATCH_FILE)
    for prev, curr in zip(fragments, fragments[1:]):
        assert curr.sentence_index >= prev.sentence_index, (
            f"sentence_index went backwards: {prev.id}={prev.sentence_index} "
            f"→ {curr.id}={curr.sentence_index}"
        )


def test_dispatch_no_leading_conjunctions():
    """No fragment should begin with a bare conjunction (and/but/or/therefore)."""
    fragments = load_text(DISPATCH_FILE)
    bad = [
        f for f in fragments
        if f.text.lower().startswith(("and ", "but ", "or ", "therefore "))
    ]
    assert bad == [], f"Fragments start with conjunctions: {[(f.id, f.text) for f in bad]}"


def test_dispatch_specific_fragments():
    """Specific expected fragments must appear in the Dispatch output."""
    fragments = load_text(DISPATCH_FILE)
    all_text = " ".join(f.text.lower() for f in fragments)

    # Core actions from the description
    assert "clarifies who will do the shipping" in all_text
    assert "package label" in all_text
    assert "picked up by the logistic company" in all_text


# ── Recourse tests ────────────────────────────────────────────────────────────


def test_recourse_minimum_fragment_count():
    """Recourse description must yield at least 8 meaningful fragments."""
    fragments = load_text(RECOURSE_FILE)
    assert len(fragments) >= 8, f"Got only {len(fragments)} fragments"


def test_recourse_invariants():
    """All recourse fragments must satisfy the shared invariants."""
    _assert_invariants(load_text(RECOURSE_FILE))


def test_recourse_contains_key_concepts():
    """Key domain concepts must appear somewhere in the Recourse fragments."""
    fragments = load_text(RECOURSE_FILE)
    all_text = " ".join(f.text for f in fragments).lower()
    assert "recourse" in all_text
    assert "collection agency" in all_text
    assert "close the case" in all_text


def test_recourse_specific_fragments():
    """Specific expected clauses must appear in the Recourse output."""
    fragments = load_text(RECOURSE_FILE)
    all_text = " ".join(f.text.lower() for f in fragments)

    assert "check" in all_text           # "check case" / "check reasoning"
    assert "payment" in all_text         # "send request for payment"
    assert "collection agency" in all_text


# ── Error handling ────────────────────────────────────────────────────────────


def test_file_not_found():
    """load_text must raise FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_text("data/nonexistent.txt")


def test_empty_file(tmp_path):
    """An empty file should return an empty fragment list without crashing."""
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    fragments = load_text(str(empty))
    assert fragments == []


def test_single_sentence(tmp_path):
    """A single short sentence should produce at least one fragment."""
    f = tmp_path / "single.txt"
    f.write_text("The secretary packages the goods carefully.", encoding="utf-8")
    fragments = load_text(str(f))
    assert len(fragments) >= 1
    assert "secretary" in fragments[0].text.lower()


def test_fragment_ids_sequential(tmp_path):
    """Fragment IDs must always be strictly sequential frag_01, frag_02, …"""
    f = tmp_path / "multi.txt"
    f.write_text(
        "The secretary clarifies the shipment method. "
        "The warehouseman packages the goods. "
        "The logistic company picks up the package.",
        encoding="utf-8",
    )
    fragments = load_text(str(f))
    for i, frag in enumerate(fragments, start=1):
        assert frag.id == f"frag_{i:02d}", f"Expected frag_{i:02d}, got {frag.id!r}"
