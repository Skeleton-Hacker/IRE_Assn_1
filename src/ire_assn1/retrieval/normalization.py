from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?", re.UNICODE)

_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

_DANISH_STOPWORDS = frozenset(
    {
        "af",
        "at",
        "de",
        "den",
        "der",
        "det",
        "en",
        "er",
        "et",
        "for",
        "fra",
        "har",
        "i",
        "ikke",
        "med",
        "og",
        "om",
        "på",
        "som",
        "til",
        "var",
        "ved",
    }
)

_STOPWORDS = {"en": _ENGLISH_STOPWORDS, "da": _DANISH_STOPWORDS}


class LanguageNormalizer:
    def __init__(self, language: str) -> None:
        normalized_language = language.casefold()
        if normalized_language not in _STOPWORDS:
            raise ValueError(f"Unsupported language: {language}")
        self.language = normalized_language
        self.stopwords = _STOPWORDS[normalized_language]

    def tokens(self, text: str) -> tuple[str, ...]:
        return tuple(
            token
            for match in _TOKEN_PATTERN.finditer(text.casefold())
            if (token := match.group()) not in self.stopwords
        )

    def normalize(self, text: str) -> str:
        return " ".join(self.tokens(text))
