import re

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e"}


def split_sentences(text: str) -> list[str]:
    """Split spoken text without breaking decimals, abbreviations, or ellipses."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    result = []
    start = 0
    for match in re.finditer(r"[.!?]+(?=\s+|$)", text):
        token = match.group(0)
        before = text[:match.start()]
        word = re.search(r"([A-Za-z.]+)$", before)
        if token == "." and (
            (word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS)
            or (match.start() > 0 and match.start() + 1 < len(text)
                and text[match.start() - 1].isdigit() and text[match.start() + 1].isdigit())
        ):
            continue
        end = match.end()
        result.append(text[start:end].strip())
        start = end
    if text[start:].strip():
        result.append(text[start:].strip())
    return result
