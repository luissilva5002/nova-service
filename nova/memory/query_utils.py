"""Shared intent gates for persistent-memory queries."""
import re


MEMORY_WRITE_QUERY = re.compile(
    r"\b(?:remember|note that|save this|save to memory|create a memory|make a memory|"
    r"store this in memory|write down)\b",
    re.I,
)

MEMORY_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "been", "being", "by", "can", "could",
    "create", "do", "does", "did", "doing", "for", "from", "good", "hello", "hey",
    "hi", "how", "i", "im", "in", "is", "it", "its", "just", "me", "memory",
    "morning", "my", "nova", "of", "on", "or", "our", "please", "remember",
    "save", "should", "so", "store", "sup", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "today", "up", "us", "was", "we",
    "what", "when", "where", "who", "why", "will", "with", "would", "write",
    "you", "your", "yours",
}


def extract_memory_keywords(query: str) -> list[str]:
    """Drop small-talk/filler tokens so irrelevant turns never trigger retrieval."""
    keywords = []
    for token in re.split(r"\W+", (query or "").strip().lower()):
        if len(token) > 2 and token not in MEMORY_STOPWORDS:
            keywords.append(token)
    return keywords
