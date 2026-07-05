import re


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))
