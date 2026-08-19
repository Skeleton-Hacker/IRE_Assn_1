from pathlib import Path

from ire_assn1.data.mind import read_mind_articles


def test_mind_articles_repair_escaped_separators(tmp_path: Path) -> None:
    path = tmp_path / "news.tsv"
    path.write_text(
        "N1\tcat\tsub\tTitle\tAbstract\\thttps://example.test/article\\t[]\t[]\n"
        "N2\tcat\tsub\tTitle 2\\tAbstract 2\thttps://example.test/article-2\t[]\t[]\n",
        encoding="utf-8",
    )

    articles = read_mind_articles(path)

    assert [(article.title, article.abstract) for article in articles] == [
        ("Title", "Abstract"),
        ("Title 2", "Abstract 2"),
    ]
