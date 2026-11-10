"""Text hygiene helpers shared by every path that feeds text to llama.cpp.

Lone UTF-16 surrogates (U+D800..U+DFFF) cannot be encoded as UTF-8. They show
up when a byte string was decoded with errors="surrogateescape" (for example a
locale-dependent name produced by the OS), and they crash tokenization with
``UnicodeEncodeError: surrogates not allowed``. ``sanitize_text`` replaces them
with U+FFFD so one bad character can never fail a turn.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("nova.text_utils")

_SURROGATES = re.compile("[\ud800-\udfff]")
_warned_sources: set[str] = set()


def has_surrogates(value: str) -> bool:
    return isinstance(value, str) and _SURROGATES.search(value) is not None


def sanitize_text(value, source: str = ""):
    """Return ``value`` with lone surrogates replaced by U+FFFD.

    Non-strings are returned unchanged. A warning naming ``source`` is logged
    once per process per source so the origin can be found without log spam.
    """
    if not isinstance(value, str):
        return value
    cleaned, count = _SURROGATES.subn("\ufffd", value)
    if count and source not in _warned_sources:
        _warned_sources.add(source)
        logger.warning(
            "Replaced %d lone surrogate character(s) in %s text; find and fix the source.",
            count,
            source or "unknown",
        )
    return cleaned
